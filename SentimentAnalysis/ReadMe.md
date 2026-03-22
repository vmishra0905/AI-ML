# Sentiment Analysis Chatbot 

An AI chatbot that analyzes the emotional sentiment of your messages in real-time and responds with empathy using OpenAI's GPT models.

## Features

- 😊 **Real-time Sentiment Analysis**: Detects positive, negative, neutral, or mixed emotions
- 🎭 **Emotion Detection**: Identifies specific emotions (happy, sad, frustrated, excited, etc.)
- 💚 **Empathetic Responses**: Responds appropriately based on your emotional state
- 📊 **Conversation Summary**: Track sentiment patterns throughout your conversation
- 🎨 **Colored Output**: Visual sentiment indicators (optional)

## Installation

### Quick Install

```bash
pip install openai --break-system-packages
```

### With Colored Output (Optional)

```bash
pip install openai colorama --break-system-packages
```

Or use the requirements file:

```bash
pip install -r sentiment_requirements.txt --break-system-packages
```

## Setup

Set your OpenAI API key:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Full Version (with sentiment logging)

```bash
python sentiment_chatbot.py
```

Features:
- Detailed sentiment analysis display
- Conversation history tracking
- Sentiment summary statistics
- Colored output (if colorama installed)

### Simple Version (minimal)

```bash
python sentiment_chatbot_simple.py
```

Features:
- Basic sentiment analysis
- Empathetic responses
- No extra dependencies
- Lighter and faster

## Example Conversation

```
You: I just got promoted at work! I'm so excited!

🔍 Analyzing sentiment...

============================================================
● Sentiment: POSITIVE
  Confidence: 95%
  Emotions: happy, excited, proud
  User expresses joy and excitement about career achievement
============================================================

💭 Thinking...

Bot: That's absolutely wonderful news! Congratulations on your promotion! 
     Your hard work and dedication have clearly paid off. How are you 
     planning to celebrate this achievement?


You: Actually, I'm a bit nervous about the new responsibilities...

🔍 Analyzing sentiment...

============================================================
● Sentiment: MIXED
  Confidence: 85%
  Emotions: nervous, anxious, uncertain, hopeful
  User shows both excitement and anxiety about change
============================================================

💭 Thinking...

Bot: It's completely natural to feel both excited and nervous about taking 
     on new responsibilities. Those feelings show that you care about doing 
     well. Remember, you were chosen for this promotion because they believe 
     in your abilities!
```

## Commands

- `quit` or `exit` - End conversation and show summary
- `summary` - Display sentiment statistics (full version only)

## How It Works

1. **User Input**: You type a message
2. **Sentiment Analysis**: OpenAI analyzes the emotional content
3. **Classification**: Detects sentiment type and specific emotions
4. **Response Generation**: Creates an empathetic response based on your mood
5. **Context Awareness**: Remembers recent conversation for better responses

## Sentiment Types

- **Positive** 😊: Happy, excited, grateful, satisfied
- **Negative** 😔: Sad, angry, frustrated, disappointed
- **Neutral** 😐: Informational, factual, calm
- **Mixed** 😕: Complex emotions (happy but nervous, sad but hopeful)

## Customization

You can modify the code to:

- Change the GPT model (`gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo`)
- Adjust response length
- Modify empathy levels
- Add custom emotion categories
- Export conversation logs

## Example Use Cases

- **Mental health check-ins**: Track your mood over time
- **Customer service**: Respond to customers based on their emotional state
- **Personal journaling**: Get empathetic feedback on your thoughts
- **Communication training**: Practice emotional awareness
- **Chatbot testing**: Test sentiment-aware conversation flows

## Tips for Best Results

- Be natural and honest in your messages
- Use complete sentences for better analysis
- The bot learns from conversation context
- Type 'summary' to see your overall mood patterns

## Limitations

- Requires OpenAI API key (costs money per message)
- Analysis quality depends on message clarity
- May not catch very subtle emotional nuances
- Rate limits apply based on your OpenAI plan

## Troubleshooting

**"Error: No API key"**: Set your OPENAI_API_KEY environment variable

**Rate limit errors**: Wait a moment or upgrade your OpenAI plan

**JSON parsing errors**: Usually temporary, try again

**Inaccurate sentiment**: Try being more explicit about your emotions

## Advanced: Programmatic Usage

```python
from sentiment_chatbot import SentimentChatbot

# Initialize
bot = SentimentChatbot()

# Analyze sentiment only
sentiment = bot.analyze_sentiment("I'm feeling great today!")
print(sentiment)
# {'sentiment': 'positive', 'confidence': 0.92, 'emotions': ['happy'], ...}

# Get response
response = bot.chat("I'm feeling great today!")
print(response)

# Get summary
summary = bot.get_sentiment_summary()
print(summary)
```

## Cost Estimation

Using `gpt-4o-mini`:
- ~$0.0002 per message pair (analysis + response)
- 100 messages ≈ $0.02
- 1000 messages ≈ $0.20

Very affordable for personal use!

## License

MIT License - Feel free to use and modify!

## Contributing

Ideas for improvements:
- Add sentiment history visualization
- Export chat logs to JSON/CSV
- Multi-language support
- Voice input/output
- Sentiment trend alerts

Feel free to fork and enhance! 🚀
