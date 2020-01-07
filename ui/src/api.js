// If the frontend is served on a non system port, we assume that the server
// is available on port 9000.
// Otherwise, we assume that the server runs on the same port as the
// frontend (80 or 443 depending on the protocol).
const protocol = window.location.protocol;
const host = window.location.hostname;
const frontendPort = window.location.port;
const protocolDefaultPort = (protocol === 'http:') ? '80' : '443';
const apiPort = (frontendPort === '') ? protocolDefaultPort : '9000';
const apiUrl = `${protocol}//${host}:${apiPort}/`;

export const getServices = fetch(apiUrl);

export const callServices = (services) => {
	var httpHeaders = new Headers();
	httpHeaders.append("Content-Type", "application/json");
	return fetch(apiUrl, {
		method: 'POST',
		body: JSON.stringify(services),
		headers: httpHeaders
	});
};