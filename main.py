from transformers import pipeline


classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)


text = input("Enter a sentence: ")


results = classifier(text)[0]


results.sort(key=lambda x: x['score'], reverse=True)

print(f"\n🧠 Detected Emotion: {results[0]['label']} (Confidence: {results[0]['score']:.2f})")
