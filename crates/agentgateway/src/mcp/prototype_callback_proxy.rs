//! PROTOTYPE, throwaway. Option 1 from nissessenap/agentgateway#1.
//!
//! When the gateway presents itself as the authorization server (rewritten `issuer`), the
//! authorization response must also come from the gateway, or RFC 9207 clients reject the
//! IdP's `iss`. So the gateway proxies the whole round trip:
//!
//! ```text
//! client ─► {meta}/authorize ─► IdP login ─► {meta}/callback ─► client redirect_uri
//! client ─► {meta}/token ─► IdP token endpoint
//! ```
//!
//! It stores nothing. The client's `redirect_uri` and `state` travel inside a sealed `state`;
//! the code handed to the client is sealed too, carrying the IdP code and the original
//! `redirect_uri`, so `/token` can enforce RFC 6749 §4.1.3 without a store.
//!
//! Keycloak only. The IdP client must list `{meta}/callback` as a valid redirect URI.

use std::sync::LazyLock;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::StatusCode;
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use http::Method;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use super::auth::{issuer_path_from_metadata_path, request_uri_for_oauth_metadata, uri_with_path};
use crate::crypto::aead::Aes256Gcm;
use crate::http::*;
use crate::proxy::ProxyError;
use crate::proxy::httpproxy::PolicyClient;
use crate::telemetry::metrics::{OutboundCallKind, OutboundCallSubtype};
use crate::types::agent::McpAuthentication;

const META_PREFIX: &str = "/.well-known/oauth-authorization-server";
const STATE_TTL_SECS: u64 = 600;
// ponytail: hardcoded; would be config on McpAuthentication.
const ALLOWED_REDIRECTS: &[&str] = &["https://claude.ai/api/mcp/auth_callback"];

// ponytail: per-process random key. Restart kills in-flight logins, and replicas can't open
// each other's state. Upgrade: key from config/secret shared across replicas.
static KEY: LazyLock<Aes256Gcm> =
	LazyLock::new(|| Aes256Gcm::new(&rand::random::<[u8; 32]>()).expect("32-byte key"));

#[derive(Serialize, Deserialize)]
struct SealedState {
	redirect_uri: String,
	state: Option<String>,
	exp: u64,
}

#[derive(Serialize, Deserialize)]
struct SealedCode {
	code: String,
	redirect_uri: String,
}

fn seal<T: Serialize>(v: &T) -> String {
	let plain = serde_json::to_vec(v).expect("serializable");
	URL_SAFE_NO_PAD.encode(KEY.seal(&plain).expect("seal"))
}

// ponytail: SealedState and SealedCode share a key; their required fields differ so one
// can't be opened as the other. Upgrade: AAD per type.
fn open<T: DeserializeOwned>(s: &str) -> Option<T> {
	let data = URL_SAFE_NO_PAD.decode(s).ok()?;
	serde_json::from_slice(&KEY.open(&data).ok()?).ok()
}

fn now() -> u64 {
	SystemTime::now()
		.duration_since(UNIX_EPOCH)
		.map(|d| d.as_secs())
		.unwrap_or_default()
}

/// The IdP only ever sees the gateway callback, so it can no longer reject unknown client
/// redirect URIs. Without this check `/authorize` is a code-theft vector: an attacker link with
/// their own `redirect_uri` and PKCE verifier gets the victim's code.
fn allowed_redirect(uri: &str) -> bool {
	let Ok(u) = url::Url::parse(uri) else {
		return false;
	};
	// RFC 8252 §7.3: loopback redirects may use any port.
	(u.scheme() == "http" && matches!(u.host_str(), Some("127.0.0.1" | "localhost" | "[::1]")))
		|| ALLOWED_REDIRECTS.contains(&uri)
}

/// External URL of the AS metadata document this endpoint hangs off, e.g.
/// `https://gw/.well-known/oauth-authorization-server/my-server/mcp`.
fn metadata_base(req: &Request, suffix: &str) -> String {
	let uri = request_uri_for_oauth_metadata(req);
	let path = uri
		.path()
		.strip_suffix(suffix)
		.unwrap_or(uri.path())
		.to_string();
	uri_with_path(uri, &path)
}

/// The issuer the gateway advertises in its metadata; must match what `authorization_server_metadata`
/// writes.
fn gateway_issuer(req: &Request, suffix: &str) -> Option<String> {
	let uri = request_uri_for_oauth_metadata(req);
	let path = uri.path().strip_suffix(suffix)?.to_string();
	let issuer_path = issuer_path_from_metadata_path(&path, META_PREFIX)?.to_string();
	Some(uri_with_path(uri, &issuer_path))
}

fn keycloak_endpoint(auth: &McpAuthentication, name: &str) -> String {
	format!("{}/protocol/openid-connect/{name}", auth.issuer)
}

fn query_pairs(req: &Request) -> Vec<(String, String)> {
	url::form_urlencoded::parse(req.uri().query().unwrap_or_default().as_bytes())
		.into_owned()
		.collect()
}

fn get<'a>(pairs: &'a [(String, String)], key: &str) -> Option<&'a str> {
	pairs
		.iter()
		.find(|(k, _)| k == key)
		.map(|(_, v)| v.as_str())
}

fn redirect(location: &str) -> Result<Response, ProxyError> {
	Ok(
		::http::Response::builder()
			.status(StatusCode::FOUND)
			.header(::http::header::LOCATION, location)
			.body(Body::empty())?,
	)
}

fn bad_request(msg: &str) -> Result<Response, ProxyError> {
	Ok(
		::http::Response::builder()
			.status(StatusCode::BAD_REQUEST)
			.body(Body::from(msg.to_string()))?,
	)
}

pub(super) async fn handle(
	req: &mut Request,
	auth: &McpAuthentication,
	client: PolicyClient,
) -> Result<Response, ProxyError> {
	let path = req.uri().path();
	if path.ends_with("/authorize") {
		authorize(req, auth)
	} else if path.ends_with("/callback") {
		callback(req, auth)
	} else {
		token(req, auth, client).await
	}
}

fn authorize(req: &Request, auth: &McpAuthentication) -> Result<Response, ProxyError> {
	let mut pairs = query_pairs(req);
	let Some(redirect_uri) = get(&pairs, "redirect_uri").map(str::to_string) else {
		return bad_request("missing redirect_uri");
	};
	// Can't redirect an error to an untrusted URI, so answer directly (RFC 6749 §4.1.2.1).
	if !allowed_redirect(&redirect_uri) {
		return bad_request("redirect_uri not allowed");
	}
	let sealed = seal(&SealedState {
		redirect_uri,
		state: get(&pairs, "state").map(str::to_string),
		exp: now() + STATE_TTL_SECS,
	});
	let callback = format!("{}/callback", metadata_base(req, "/authorize"));
	pairs.retain(|(k, _)| k != "redirect_uri" && k != "state");
	let query = url::form_urlencoded::Serializer::new(String::new())
		.extend_pairs(pairs)
		.append_pair("redirect_uri", &callback)
		.append_pair("state", &sealed)
		.finish();
	redirect(&format!("{}?{query}", keycloak_endpoint(auth, "auth")))
}

fn callback(req: &Request, auth: &McpAuthentication) -> Result<Response, ProxyError> {
	let pairs = query_pairs(req);
	// The gateway is Keycloak's client, so it runs the RFC 9207 check itself.
	if let Some(iss) = get(&pairs, "iss")
		&& iss != auth.issuer
	{
		return bad_request("iss does not match the configured issuer");
	}
	let Some(st) = get(&pairs, "state").and_then(open::<SealedState>) else {
		return bad_request("invalid state");
	};
	if now() > st.exp {
		return bad_request("login expired");
	}
	let Some(issuer) = gateway_issuer(req, "/callback") else {
		return bad_request("cannot derive issuer");
	};
	let Ok(mut location) = url::Url::parse(&st.redirect_uri) else {
		return bad_request("invalid redirect_uri");
	};
	{
		let mut q = location.query_pairs_mut();
		if let Some(code) = get(&pairs, "code") {
			q.append_pair(
				"code",
				&seal(&SealedCode {
					code: code.to_string(),
					redirect_uri: st.redirect_uri.clone(),
				}),
			);
		}
		// Error responses need the same treatment, or clients report an issuer mismatch
		// instead of e.g. access_denied.
		for k in ["error", "error_description", "error_uri"] {
			if let Some(v) = get(&pairs, k) {
				q.append_pair(k, v);
			}
		}
		if let Some(s) = &st.state {
			q.append_pair("state", s);
		}
		q.append_pair("iss", &issuer);
	}
	redirect(location.as_str())
}

/// Swap the wrapped code for the IdP's and the client redirect_uri for the gateway callback.
/// Other grants (refresh_token) pass through untouched.
fn rewrite_token_form(form: &[u8], callback: &str) -> Result<String, &'static str> {
	let pairs: Vec<(String, String)> = url::form_urlencoded::parse(form).into_owned().collect();
	if get(&pairs, "grant_type") != Some("authorization_code") {
		return Ok(String::from_utf8_lossy(form).into_owned());
	}
	let sc = get(&pairs, "code")
		.and_then(open::<SealedCode>)
		.ok_or("code was not issued by this gateway")?;
	if get(&pairs, "redirect_uri") != Some(sc.redirect_uri.as_str()) {
		return Err("redirect_uri does not match the authorization request");
	}
	let mut out = url::form_urlencoded::Serializer::new(String::new());
	for (k, v) in &pairs {
		match k.as_str() {
			"code" => out.append_pair(k, &sc.code),
			"redirect_uri" => out.append_pair(k, callback),
			_ => out.append_pair(k, v),
		};
	}
	Ok(out.finish())
}

async fn token(
	req: &mut Request,
	auth: &McpAuthentication,
	client: PolicyClient,
) -> Result<Response, ProxyError> {
	if req.method() != Method::POST {
		return Ok(
			::http::Response::builder()
				.status(StatusCode::METHOD_NOT_ALLOWED)
				.header(::http::header::ALLOW, "POST")
				.body(Body::empty())?,
		);
	}
	let callback = format!("{}/callback", metadata_base(req, "/token"));
	let authorization = req.headers().get(::http::header::AUTHORIZATION).cloned();
	let limit = crate::http::buffer_limit(req);
	let body = std::mem::take(req.body_mut());
	let bytes = crate::http::read_body_with_limit(body, limit)
		.await
		.map_err(ProxyError::Body)?;
	let form = match rewrite_token_form(&bytes, &callback) {
		Ok(f) => f,
		Err(why) => {
			let body = serde_json::json!({"error": "invalid_grant", "error_description": why});
			return Ok(
				::http::Response::builder()
					.status(StatusCode::BAD_REQUEST)
					.header(::http::header::CONTENT_TYPE, "application/json")
					.body(Body::from(body.to_string()))?,
			);
		},
	};
	let mut builder = ::http::Request::builder()
		.uri(keycloak_endpoint(auth, "token"))
		.method(Method::POST)
		.header(
			::http::header::CONTENT_TYPE,
			"application/x-www-form-urlencoded",
		);
	if let Some(a) = authorization {
		builder = builder.header(::http::header::AUTHORIZATION, a);
	}
	client
		.with_outbound(OutboundCallKind::Policy, OutboundCallSubtype::Oidc)
		.simple_call(builder.body(Body::from(form))?)
		.await
}

#[cfg(test)]
mod tests {
	use super::*;

	const META: &str = "https://gw.example.com/.well-known/oauth-authorization-server/my-server/mcp";
	const CLIENT_CB: &str = "http://127.0.0.1:53682/callback";

	fn auth() -> McpAuthentication {
		let mut a = super::super::auth::tests::default_auth();
		a.issuer = "https://idp.example.com/realms/example".to_string();
		a.provider = Some(crate::types::agent::McpIDP::Keycloak {});
		a
	}

	fn get_req(uri: &str) -> Request {
		::http::Request::builder()
			.uri(uri)
			.body(Body::empty())
			.unwrap()
	}

	fn location(resp: &Response) -> url::Url {
		assert_eq!(resp.status(), StatusCode::FOUND);
		url::Url::parse(resp.headers()[::http::header::LOCATION].to_str().unwrap()).unwrap()
	}

	fn param(u: &url::Url, k: &str) -> Option<String> {
		u.query_pairs()
			.find(|(kk, _)| kk == k)
			.map(|(_, v)| v.into_owned())
	}

	fn authorize_with(redirect_uri: &str) -> Response {
		let q = url::form_urlencoded::Serializer::new(String::new())
			.append_pair("response_type", "code")
			.append_pair("client_id", "mcp-gateway")
			.append_pair("redirect_uri", redirect_uri)
			.append_pair("state", "client-state")
			.append_pair("code_challenge", "abc")
			.finish();
		authorize(&get_req(&format!("{META}/authorize?{q}")), &auth()).unwrap()
	}

	/// Full browser round trip: client → gateway → Keycloak → gateway → client.
	#[test]
	fn round_trip_gives_client_gateway_iss_and_its_own_state() {
		let to_idp = location(&authorize_with(CLIENT_CB));
		assert_eq!(
			to_idp.as_str().split('?').next().unwrap(),
			"https://idp.example.com/realms/example/protocol/openid-connect/auth"
		);
		assert_eq!(
			param(&to_idp, "redirect_uri").unwrap(),
			format!("{META}/callback")
		);
		assert_eq!(param(&to_idp, "code_challenge").unwrap(), "abc");

		// Keycloak redirects the browser back to the gateway callback.
		let q = url::form_urlencoded::Serializer::new(String::new())
			.append_pair("code", "kc-code")
			.append_pair("state", &param(&to_idp, "state").unwrap())
			.append_pair("iss", "https://idp.example.com/realms/example")
			.finish();
		let to_client =
			location(&callback(&get_req(&format!("{META}/callback?{q}")), &auth()).unwrap());

		assert_eq!(to_client.as_str().split('?').next().unwrap(), CLIENT_CB);
		assert_eq!(param(&to_client, "state").unwrap(), "client-state");
		// Matches the issuer the metadata advertises, so RFC 9207 clients accept it.
		assert_eq!(
			param(&to_client, "iss").unwrap(),
			"https://gw.example.com/my-server/mcp"
		);

		// Token step: wrapped code + client redirect_uri → Keycloak code + gateway callback.
		let form = url::form_urlencoded::Serializer::new(String::new())
			.append_pair("grant_type", "authorization_code")
			.append_pair("code", &param(&to_client, "code").unwrap())
			.append_pair("redirect_uri", CLIENT_CB)
			.append_pair("code_verifier", "v")
			.finish();
		let out = rewrite_token_form(form.as_bytes(), &format!("{META}/callback")).unwrap();
		let out: Vec<_> = url::form_urlencoded::parse(out.as_bytes())
			.into_owned()
			.collect();
		assert_eq!(get(&out, "code"), Some("kc-code"));
		assert_eq!(
			get(&out, "redirect_uri"),
			Some(format!("{META}/callback").as_str())
		);
		assert_eq!(get(&out, "code_verifier"), Some("v"));
	}

	#[test]
	fn authorize_rejects_non_loopback_redirect() {
		assert_eq!(
			authorize_with("https://evil.example/steal").status(),
			StatusCode::BAD_REQUEST
		);
		assert_eq!(
			location(&authorize_with("http://localhost:1/cb")).scheme(),
			"https"
		);
		assert_eq!(
			location(&authorize_with("https://claude.ai/api/mcp/auth_callback")).scheme(),
			"https"
		);
	}

	#[test]
	fn callback_forwards_errors_with_gateway_iss() {
		let st = seal(&SealedState {
			redirect_uri: CLIENT_CB.into(),
			state: Some("s".into()),
			exp: now() + 60,
		});
		let q = format!(
			"error=access_denied&state={st}&iss=https%3A%2F%2Fidp.example.com%2Frealms%2Fexample"
		);
		let u = location(&callback(&get_req(&format!("{META}/callback?{q}")), &auth()).unwrap());
		assert_eq!(param(&u, "error").unwrap(), "access_denied");
		assert_eq!(
			param(&u, "iss").unwrap(),
			"https://gw.example.com/my-server/mcp"
		);
		assert_eq!(param(&u, "code"), None);
	}

	#[test]
	fn callback_rejects_tampered_expired_or_foreign_iss() {
		let ok = seal(&SealedState {
			redirect_uri: CLIENT_CB.into(),
			state: None,
			exp: now() + 60,
		});
		let expired = seal(&SealedState {
			redirect_uri: CLIENT_CB.into(),
			state: None,
			exp: 0,
		});
		let code = seal(&SealedCode {
			code: "c".into(),
			redirect_uri: CLIENT_CB.into(),
		});
		for q in [
			format!("code=c&state={ok}x"),
			format!("code=c&state={expired}"),
			format!("code=c&state={code}"),
			format!("code=c&state={ok}&iss=https%3A%2F%2Fother.example"),
		] {
			let resp = callback(&get_req(&format!("{META}/callback?{q}")), &auth()).unwrap();
			assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{q}");
		}
	}

	#[test]
	fn token_rejects_mismatched_redirect_and_unwrapped_code() {
		let code = seal(&SealedCode {
			code: "c".into(),
			redirect_uri: CLIENT_CB.into(),
		});
		let cb = format!("{META}/callback");
		let form = format!(
			"grant_type=authorization_code&code={code}&redirect_uri=http%3A%2F%2F127.0.0.1%3A9999%2Fcb"
		);
		assert!(rewrite_token_form(form.as_bytes(), &cb).is_err());
		assert!(rewrite_token_form(b"grant_type=authorization_code&code=raw-kc-code", &cb).is_err());
		assert_eq!(
			rewrite_token_form(b"grant_type=refresh_token&refresh_token=r", &cb).unwrap(),
			"grant_type=refresh_token&refresh_token=r"
		);
	}
}
