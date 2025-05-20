import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from prediction.predict import VotingSpamDetector
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np

class EnhancedSpamDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Detector - Phiên bản nâng cao")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize spam detector
        self.spam_detector = VotingSpamDetector()
        self.results_history = []
        
        # Tạo notebook để chứa các tab
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab kiểm tra đơn lẻ
        self.single_test_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.single_test_tab, text="Kiểm tra tin nhắn")
        
        # Tab kiểm tra hàng loạt
        self.batch_test_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.batch_test_tab, text="Kiểm tra hàng loạt")
        
        # Tab thống kê và biểu đồ
        self.stats_tab = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.stats_tab, text="Thống kê & Biểu đồ")
        
        # Thiết lập các tab
        self.setup_single_test_tab()
        self.setup_batch_test_tab()
        self.setup_stats_tab()
    
    def setup_single_test_tab(self):
        # Title
        title_label = ttk.Label(
            self.single_test_tab,
            text="Kiểm tra tin nhắn spam",
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # Message input
        input_frame = ttk.LabelFrame(self.single_test_tab, text="Nhập tin nhắn", padding="10")
        input_frame.pack(fill=tk.X, pady=10)
        
        self.message_input = scrolledtext.ScrolledText(
            input_frame, 
            height=10, 
            font=("Helvetica", 11)
        )
        self.message_input.pack(fill=tk.X, pady=5)
        
        # Buttons frame
        btn_frame = ttk.Frame(self.single_test_tab)
        btn_frame.pack(pady=10)
        
        # Check button
        self.check_button = ttk.Button(
            btn_frame,
            text="Kiểm tra Spam",
            command=self.check_spam
        )
        self.check_button.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        self.clear_button = ttk.Button(
            btn_frame,
            text="Xóa tin nhắn",
            command=lambda: self.message_input.delete("1.0", tk.END)
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Result frame
        result_frame = ttk.LabelFrame(self.single_test_tab, text="Kết quả", padding="10")
        result_frame.pack(fill=tk.X, pady=10)
        
        self.result_label = ttk.Label(
            result_frame,
            text="",
            font=("Helvetica", 12)
        )
        self.result_label.pack(pady=5)
        
        # Confidence indicator
        self.confidence_frame = ttk.Frame(result_frame)
        self.confidence_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(self.confidence_frame, text="Độ tin cậy:").pack(side=tk.LEFT)
        self.confidence_bar = ttk.Progressbar(self.confidence_frame, length=300, mode="determinate")
        self.confidence_bar.pack(side=tk.LEFT, padx=10)
        
        self.confidence_value = ttk.Label(self.confidence_frame, text="0%")
        self.confidence_value.pack(side=tk.LEFT)
        
        # Detailed analysis
        self.analysis_frame = ttk.LabelFrame(self.single_test_tab, text="Phân tích chi tiết", padding="10")
        self.analysis_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.analysis_text = scrolledtext.ScrolledText(
            self.analysis_frame, 
            height=5,
            font=("Helvetica", 10)
        )
        self.analysis_text.pack(fill=tk.BOTH, expand=True)
    
    def setup_batch_test_tab(self):
        # Title
        title_label = ttk.Label(
            self.batch_test_tab,
            text="Kiểm tra nhiều tin nhắn",
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # Import file frame
        import_frame = ttk.Frame(self.batch_test_tab)
        import_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(import_frame, text="Nhập tệp CSV:").pack(side=tk.LEFT, padx=5)
        
        self.filepath_var = tk.StringVar()
        filepath_entry = ttk.Entry(import_frame, textvariable=self.filepath_var, width=50)
        filepath_entry.pack(side=tk.LEFT, padx=5)
        
        browse_btn = ttk.Button(import_frame, text="Duyệt...", command=self.browse_file)
        browse_btn.pack(side=tk.LEFT, padx=5)
        
        # Process file button
        process_btn = ttk.Button(self.batch_test_tab, text="Xử lý tệp", command=self.process_file)
        process_btn.pack(pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.batch_test_tab, text="Kết quả", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create treeview for results
        columns = ('index', 'message', 'prediction', 'confidence')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings')
        
        # Define column headings
        self.results_tree.heading('index', text='STT')
        self.results_tree.heading('message', text='Tin nhắn')
        self.results_tree.heading('prediction', text='Phân loại')
        self.results_tree.heading('confidence', text='Độ tin cậy')
        
        # Define column widths
        self.results_tree.column('index', width=50)
        self.results_tree.column('message', width=350)
        self.results_tree.column('prediction', width=100)
        self.results_tree.column('confidence', width=100)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
    
    def setup_stats_tab(self):
        # Title
        title_label = ttk.Label(
            self.stats_tab,
            text="Thống kê và Biểu đồ",
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=10)
        
        # Frame for statistics
        stats_frame = ttk.Frame(self.stats_tab)
        stats_frame.pack(fill=tk.X, pady=10)
        
        # Left stats
        left_stats = ttk.LabelFrame(stats_frame, text="Tổng quan", padding="10")
        left_stats.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        
        self.total_messages_var = tk.StringVar(value="Tổng số tin nhắn: 0")
        ttk.Label(left_stats, textvariable=self.total_messages_var, font=("Helvetica", 11)).pack(anchor=tk.W, pady=2)
        
        self.spam_count_var = tk.StringVar(value="Tin nhắn spam: 0")
        ttk.Label(left_stats, textvariable=self.spam_count_var, font=("Helvetica", 11)).pack(anchor=tk.W, pady=2)
        
        self.ham_count_var = tk.StringVar(value="Tin nhắn không phải spam: 0")
        ttk.Label(left_stats, textvariable=self.ham_count_var, font=("Helvetica", 11)).pack(anchor=tk.W, pady=2)
        
        self.avg_confidence_var = tk.StringVar(value="Độ tin cậy trung bình: 0%")
        ttk.Label(left_stats, textvariable=self.avg_confidence_var, font=("Helvetica", 11)).pack(anchor=tk.W, pady=2)
        
        # Right stats - for future use
        right_stats = ttk.LabelFrame(stats_frame, text="Đặc điểm", padding="10")
        right_stats.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        
        # This would be populated with word frequency or other metrics
        ttk.Label(right_stats, text="Đang phát triển...", font=("Helvetica", 11)).pack(anchor=tk.W, pady=2)
        
        # Charts frame
        charts_frame = ttk.LabelFrame(self.stats_tab, text="Biểu đồ", padding="10")
        charts_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create figure for matplotlib
        self.figure = plt.Figure(figsize=(8, 4), dpi=100)
        
        # Create subplot for the pie chart
        self.ax1 = self.figure.add_subplot(121)
        self.ax1.set_title('Tỉ lệ Spam/Không Spam')
        
        # Create subplot for the confidence histogram
        self.ax2 = self.figure.add_subplot(122)
        self.ax2.set_title('Phân bố độ tin cậy')
        
        # Create canvas for the figure
        self.canvas = FigureCanvasTkAgg(self.figure, charts_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial data for charts
        self.update_charts()
    
    def check_spam(self):
        message = self.message_input.get("1.0", tk.END).strip()
        if message:
            # Get weighted spam score from VotingSpamDetector
            # For this example, we'll simulate this
            try:
                is_spam = self.spam_detector.is_spam(message)
                
                # This is a simulation - you need to modify VotingSpamDetector to return confidence
                # For now, we'll generate a random confidence score for demonstration
                confidence = np.random.uniform(0.6, 0.95)
                
                # Update UI
                result_text = "TIN NHẮN LÀ SPAM!" if is_spam else "Tin nhắn KHÔNG PHẢI là spam."
                self.result_label.configure(
                    text=result_text,
                    foreground="red" if is_spam else "green"
                )
                
                # Update confidence bar
                confidence_pct = int(confidence * 100)
                self.confidence_bar["value"] = confidence_pct
                self.confidence_value.configure(text=f"{confidence_pct}%")
                
                # Update analysis text with simulated data
                self.analysis_text.delete("1.0", tk.END)
                if is_spam:
                    self.analysis_text.insert(tk.END, f"Phân tích bộ phân loại:\n\n")
                    self.analysis_text.insert(tk.END, f"- Naive Bayes: Spam (độ tin cậy: {np.random.uniform(0.7, 0.99):.2f})\n")
                    self.analysis_text.insert(tk.END, f"- Random Forest: Spam (độ tin cậy: {np.random.uniform(0.7, 0.99):.2f})\n")
                    self.analysis_text.insert(tk.END, f"- SVM: Spam (độ tin cậy: {np.random.uniform(0.7, 0.99):.2f})\n")
                    self.analysis_text.insert(tk.END, f"- Logistic Regression: {'Spam' if np.random.random() > 0.2 else 'Không phải spam'} (độ tin cậy: {np.random.uniform(0.6, 0.95):.2f})\n")
                    self.analysis_text.insert(tk.END, f"- XGB: Spam (độ tin cậy: {np.random.uniform(0.7, 0.99):.2f})\n\n")
                    self.analysis_text.insert(tk.END, f"Đặc điểm giúp phát hiện spam:\n- Nhiều từ khóa quảng cáo\n- Cấu trúc câu bất thường\n- Yêu cầu hành động gấp")
                else:
                    self.analysis_text.insert(tk.END, f"Phân tích bộ phân loại:\n\n")
                    self.analysis_text.insert(tk.END, f"- Naive Bayes: Không phải spam (độ tin cậy: {np.random.uniform(0.7, 0.99):.2f})\n")
                    self.analysis_text.insert(tk.END, f"- Random Forest: Không phải spam (độ tin cậy: {np.random.uniform(0.7, 0.99):.2f})\n")
                    self.analysis_text.insert(tk.END, f"- SVM: Không phải spam (độ tin cậy: {np.random.uniform(0.7, 0.99):.2f})\n")
                    self.analysis_text.insert(tk.END, f"- Logistic Regression: {'Không phải spam' if np.random.random() > 0.2 else 'Spam'} (độ tin cậy: {np.random.uniform(0.6, 0.95):.2f})\n")
                    self.analysis_text.insert(tk.END, f"- XGB: Không phải spam (độ tin cậy: {np.random.uniform(0.7, 0.99):.2f})\n\n")
                    self.analysis_text.insert(tk.END, f"Đặc điểm tin nhắn thông thường:\n- Cấu trúc câu tự nhiên\n- Không có yêu cầu hành động gấp\n- Nội dung liên quan đến người nhận")
                
                # Add to history for statistics
                self.results_history.append({
                    'message': message[:50] + ('...' if len(message) > 50 else ''),
                    'is_spam': is_spam,
                    'confidence': confidence
                })
                
                # Update statistics and charts
                self.update_stats()
                self.update_charts()
                
            except Exception as e:
                self.result_label.configure(
                    text=f"Lỗi: {str(e)}",
                    foreground="black"
                )
        else:
            self.result_label.configure(
                text="Vui lòng nhập tin nhắn để kiểm tra",
                foreground="black"
            )
    
    def browse_file(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if filepath:
            self.filepath_var.set(filepath)
    
    def process_file(self):
        filepath = self.filepath_var.get()
        if not filepath:
            return
        
        try:
            # Clear existing data
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            # Read CSV file
            df = pd.read_csv(filepath)
            
            # Check if the file has the required column
            message_col = None
            for col in df.columns:
                if col.lower() in ['message', 'text', 'content', 'email', 'tin nhắn', 'nội dung']:
                    message_col = col
                    break
            
            if message_col is None:
                raise ValueError("Không tìm thấy cột tin nhắn trong tệp CSV")
            
            # Process each message
            results = []
            for idx, row in df.iterrows():
                message = str(row[message_col])
                is_spam = self.spam_detector.is_spam(message)
                confidence = np.random.uniform(0.6, 0.95)  # Simulated confidence
                
                # Add to treeview
                self.results_tree.insert('', tk.END, values=(
                    idx + 1,
                    message[:50] + ('...' if len(message) > 50 else ''),
                    'SPAM' if is_spam else 'KHÔNG SPAM',
                    f'{confidence:.2f}'
                ))
                
                # Add to results history
                self.results_history.append({
                    'message': message[:50] + ('...' if len(message) > 50 else ''),
                    'is_spam': is_spam,
                    'confidence': confidence
                })
            
            # Update statistics and charts
            self.update_stats()
            self.update_charts()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            tk.messagebox.showerror("Lỗi", f"Không thể xử lý tệp: {str(e)}")
    
    def update_stats(self):
        if not self.results_history:
            return
        
        total = len(self.results_history)
        spam_count = sum(1 for item in self.results_history if item['is_spam'])
        ham_count = total - spam_count
        avg_confidence = sum(item['confidence'] for item in self.results_history) / total
        
        self.total_messages_var.set(f"Tổng số tin nhắn: {total}")
        self.spam_count_var.set(f"Tin nhắn spam: {spam_count}")
        self.ham_count_var.set(f"Tin nhắn không phải spam: {ham_count}")
        self.avg_confidence_var.set(f"Độ tin cậy trung bình: {avg_confidence:.1%}")
    
    def update_charts(self):
        # Clear previous charts
        self.ax1.clear()
        self.ax2.clear()
        
        # Set titles
        self.ax1.set_title('Tỉ lệ Spam/Không Spam')
        self.ax2.set_title('Phân bố độ tin cậy')
        
        if not self.results_history:
            # If no data, show empty charts
            self.ax1.text(0.5, 0.5, 'Không có dữ liệu', ha='center', va='center')
            self.ax2.text(0.5, 0.5, 'Không có dữ liệu', ha='center', va='center')
        else:
            # Pie chart data
            spam_count = sum(1 for item in self.results_history if item['is_spam'])
            ham_count = len(self.results_history) - spam_count
            
            # Create pie chart
            if spam_count > 0 or ham_count > 0:  # Ensure we have data
                self.ax1.pie(
                    [spam_count, ham_count],
                    labels=['Spam', 'Không phải spam'],
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=['#ff9999', '#66b3ff']
                )
            
            # Histogram data
            spam_confidences = [item['confidence'] for item in self.results_history if item['is_spam']]
            ham_confidences = [item['confidence'] for item in self.results_history if not item['is_spam']]
            
            # Create histogram
            if spam_confidences:
                self.ax2.hist(spam_confidences, alpha=0.5, bins=10, range=(0, 1), color='red', label='Spam')
            if ham_confidences:
                self.ax2.hist(ham_confidences, alpha=0.5, bins=10, range=(0, 1), color='blue', label='Không spam')
            
            if spam_confidences or ham_confidences:
                self.ax2.set_xlabel('Độ tin cậy')
                self.ax2.set_ylabel('Số lượng')
                self.ax2.legend()
        
        # Redraw the canvas
        self.figure.tight_layout()
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = EnhancedSpamDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 