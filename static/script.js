document.getElementById("summaryForm")?.addEventListener("submit", async (event) => {
    event.preventDefault(); // Prevent page reload

    const youtube_url = document.getElementById("youtube_url").value;

    try {
        const response = await fetch("/page1", {
            method: "POST", // Use POST method
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ youtube_url }),
        });

        const data = await response.json();
        const summaryElement = document.getElementById("summary");
        summaryElement.style.display = "block";
        summaryElement.innerHTML = data.summary || `<strong>Error:</strong> ${data.error}`;
    } catch (error) {
        console.error("Error fetching summary:", error);
        alert("An error occurred while fetching the summary.");
    }
});


document.getElementById("chatbotForm")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const youtube_url = document.getElementById("youtube_url").value;
    const question = document.getElementById("question").value;
    const response = await fetch("/page2", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ youtube_url, question }),
    });
    const data = await response.json();
    document.getElementById("response").innerHTML = data.answer || data.error;
});
