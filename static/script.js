function analyzeTweet() {
    var tweet = document.getElementById("tweet-input").value;
    fetch("/analyze", {
        method: "POST",
        body: JSON.stringify({ tweet: tweet }),
        headers: {
            "Content-Type": "application/json"
        }
    })
    .then(response => response.json())
    .then(data => {
        var resultDiv = document.getElementById("result");
        resultDiv.innerHTML = `
            <p>Original Tweet: ${data.tweet}</p>
            <p>BERT Sentiment: ${data.bert_sentiment}</p>
            <p>Zero-Shot Classification: ${data.zero_shot_classification}</p>
            <p>Combined Sentiment: ${data.combined_sentiment}</p>
            <p>BERT Sentiment Probability: ${data.bert_prob}</p>
            <p>Zero-Shot Classification Probability: ${data.zero_shot_prob}</p>
        `;
    });
}
    