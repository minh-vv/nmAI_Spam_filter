from spam_detector_ai.prediction.predict import VotingSpamDetector

# Tạo đối tượng detector
spam_detector = VotingSpamDetector()

# Dự đoán tin nhắn
message = "Come pick me up tomorrow"
is_spam = spam_detector.is_spam(message)
print(f"Tin nhắn: {message}")
print(f"Is spam: {is_spam}")