export async function fetchJSON(url) {
    const response = await fetch(url);

    if (!response.ok) {
        alert("Request failed: " + url);
        return null;
    }

    return response.json();
}

export function setHTML(element, html) {
    element.innerHTML = html;
}