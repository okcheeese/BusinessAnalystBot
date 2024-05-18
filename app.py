from flask import Flask, render_template, request
import google.generativeai as genai
from nltk.sentiment import SentimentIntensityAnalyzer
import sys
from nltk import word_tokenize, pos_tag, ne_chunk

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():

    genai.configure(api_key="YOUR_API_KEY")

    # Set up the model
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
    }

    safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    ]

    system_instruction = "Act as a seasoned business advisor, assisting an entrepreneur in developing a robust business plan. Offer guidance on conducting thorough market analysis to identify opportunities and challenges. Help define the target audience and devise effective strategies for customer acquisition and retention. Explore various revenue models, such as subscription-based, freemium, or e-commerce, and advise on selecting the most suitable one. Discuss scalability options, including potential partnerships, expansion into new markets, or diversification of product offerings. Analyze relevant competitors and provide actionable insights to ensure the business plan is strategic, practical, and aligned with the entrepreneur's long-term objectives. Add <br> after line breaks and <n> after new paragraphs/lines"

    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                generation_config=generation_config,
                                system_instruction=system_instruction,
                                safety_settings=safety_settings)

    convo = model.start_chat(history=[
    ])


    def get_entities(text):
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        chunks = ne_chunk(pos_tags)

        proper_nouns = []

        for chunk in chunks:
            if hasattr(chunk, 'label') and chunk.label() in ['PERSON', 'ORGANIZATION', 'GPE']:
                proper_nouns.append(' '.join(c[0] for c in chunk))

        return proper_nouns

    def get_sentiment(text):
        sia = SentimentIntensityAnalyzer()
        sentiment_score = sia.polarity_scores(text)
        return sentiment_score['compound']

    # def get_topics(text):
    #   stop_words = nltk.corpus.stopwords.words('english')  

    #   processed_text = [word for word in word_tokenize(text.lower()) if word not in stop_words]

    #   documents = [processed_text]  
    #   dictionary = corpora.Dictionary(documents)

    #   corpus = [dictionary.doc2bow(doc) for doc in documents]

    #   num_topics = 3  
    #   lda_model = LdaModel(corpus, id2word=dictionary, num_topics=num_topics)

    #   topics = lda_model.show_topic(0, topn=2)  
    #   topic_descriptions = [f"{topic[1]:.2f} - {', '.join(word for word, score in topics)}" for topic in topics]
    #   return topic_descriptions

    whatever = request.args.get('msg')
    entities = get_entities(whatever)
    sentiment = get_sentiment(whatever)

# Update convo message with extracted information
    convo.send_message(f"{whatever} (Entities: {entities}, Sentiment: {sentiment})")
    return str(convo.last.text)

app.run(debug = True)
