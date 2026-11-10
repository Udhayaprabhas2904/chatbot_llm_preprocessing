
from chatbots.baseline_chatbot import BaselineChatbot
from chatbots.preprocessing_chatbot import PreprocessingChatbot


class FakeClient:
    def generate(self, model, prompt, **kwargs):
        return {
            "response": f"Answer for: {prompt}",
            "processing_time_seconds": 0.1,
        }


def test_baseline_chatbot_uses_original_query():
    bot = BaselineChatbot(client=FakeClient())
    result = bot.respond("  Hello   world  ", "test-model")

    assert result["response"]
    assert result["submitted_prompt"] == "  Hello   world  "


def test_preprocessing_chatbot_uses_processed_query():
    bot = PreprocessingChatbot(client=FakeClient())
    result = bot.respond("  Hello   world  ", "test-model")

    assert result["response"]
    assert result["submitted_prompt"] != "  Hello   world  "
