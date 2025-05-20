import tkinter as tk
from tkinter import ttk
from prediction.predict import VotingSpamDetector

class SpamDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Detector")
        self.root.geometry("600x400")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize spam detector
        self.spam_detector = VotingSpamDetector()
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            self.main_frame,
            text="Spam Message Detector",
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # Message input
        input_frame = ttk.LabelFrame(self.main_frame, text="Enter Message", padding="10")
        input_frame.pack(fill=tk.X, pady=10)
        
        self.message_input = tk.Text(input_frame, height=4, width=50, font=("Helvetica", 11))
        self.message_input.pack(fill=tk.X, pady=5)
        
        # Check button
        self.check_button = ttk.Button(
            self.main_frame,
            text="Check for Spam",
            command=self.check_spam
        )
        self.check_button.pack(pady=10)
        
        # Result frame
        result_frame = ttk.LabelFrame(self.main_frame, text="Result", padding="10")
        result_frame.pack(fill=tk.X, pady=10)
        
        self.result_label = ttk.Label(
            result_frame,
            text="",
            font=("Helvetica", 11)
        )
        self.result_label.pack(pady=5)
        
    def check_spam(self):
        message = self.message_input.get("1.0", tk.END).strip()
        if message:
            is_spam = self.spam_detector.is_spam(message)
            result_text = "This message is SPAM!" if is_spam else "This message is NOT spam."
            self.result_label.configure(
                text=result_text,
                foreground="red" if is_spam else "green"
            )
        else:
            self.result_label.configure(
                text="Please enter a message to check",
                foreground="black"
            )

def main():
    root = tk.Tk()
    app = SpamDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 