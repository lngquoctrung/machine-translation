import time
import sys
import gradio as gr
import tensorflow as tf
from pathlib import Path

root_dir = str(Path(__file__).parent.parent.absolute())
if not root_dir in sys.path:
    sys.path.insert(0, root_dir)

from config import Config
from src.evaluation.metrics import BLEUScore
from src.utils import load_tokenizer
from src.evaluation.beam_search import BeamSearchDecoder

class TranslationApp:
    """Gradio translation application"""

    def __init__(self, model_path: str, tokenizer_path: str, config: dict):
        print("Loading model...")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.tokenizer_src = load_tokenizer(f"{tokenizer_path}/tokenizer_en.pkl")
        self.tokenizer_trg = load_tokenizer(f"{tokenizer_path}/tokenizer_vi.pkl")
        self.config = config
        self.bleu_scorer = BLEUScore()
        self.history = []
        self.decoder = BeamSearchDecoder(
            self.model,
            self.tokenizer_src,
            self.tokenizer_trg,
            config.MAX_LENGTH_SRC,
            config.MAX_LENGTH_TRG,
            beam_width=config.BEAM_WIDTH
        )
        print("Model loaded!")

    def translate(self, text: str, reference: str = "") -> tuple:
        """Main translation function"""
        if not text.strip():
            return "", "Please enter text to translate"

        start_time = time.time()
        translation = self.decoder.decode_greedy(text)
        method = "Greedy Decoding"
        time_taken = time.time() - start_time

        bleu_info = ""
        if reference.strip():
            bleu = self.bleu_scorer.compute(reference.lower(), translation.lower())
            bleu_info = f"\n- **BLEU Score:** {bleu:.2f}"

        info = f"""
            **Translation Info:**
            - **Method:** {method}
            - **Time:** {time_taken:.2f}s
            - **Input:** {len(text.split())} words
            - **Output:** {len(translation.split())} words{bleu_info}
        """.strip()

        self.history.append({
            'input': text,
            'output': translation,
            'method': method,
            'time': time_taken
        })

        return translation, info

    def get_history(self) -> str:
        """Get translation history"""
        if not self.history:
            return "No translation history yet."

        table = "| # | Input | Translation | Time |\n"
        table += "|---|-------|-------------|------|\n"

        for i, item in enumerate(reversed(self.history[-10:]), 1):
            inp = item['input'][:40] + "..." if len(item['input']) > 40 else item['input']
            out = item['output'][:40] + "..." if len(item['output']) > 40 else item['output']
            table += f"| {i} | {inp} | {out} | {item['time']:.2f}s |\n"

        return table

    def clear_history(self) -> str:
        """Clear history"""
        self.history = []
        return "History cleared!"

    def launch(self, share=True):
        """Launch Gradio app"""

        examples = [
            ["Hello, how are you?", ""],
            ["I love machine learning.", ""],
            ["The weather is beautiful today.", ""],
            ["Thank you for your help.", ""],
        ]

        # Minimal CSS - chỉ font
        custom_css = """
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }
        """

        with gr.Blocks(
            title="English-Vietnamese Translation", 
            theme=gr.themes.Soft(
                primary_hue="purple",
                secondary_hue="blue"
            ),
            css=custom_css
        ) as demo:

            gr.Markdown("""
            # English to Vietnamese Translation
            ## BiLSTM + Multi-Head Attention Model
            """)

            # Pure Markdown Disclaimer - NO HTML
            gr.Markdown("""
                > ### Experimental Project Notice
                > 
                > **This is an experimental research project.** The model has the following limitations:
                > 
                > - Works best with **short, simple sentences** (less than 15 words)
                > - May struggle with **complex sentence structures**
                > - **Limited vocabulary** (30k English, 5k Vietnamese words)
                > - Proper names may not translate correctly (uses copy-through mechanism)
                > - Not suitable for production use
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input")

                    input_text = gr.Textbox(
                        label="English Text",
                        placeholder="Enter English text here... (Keep it short and simple for best results)",
                        lines=6,
                        max_lines=10
                    )

                    reference_text = gr.Textbox(
                        label="Reference Translation (Optional - for BLEU score)",
                        placeholder="Vietnamese reference translation...",
                        lines=3
                    )

                    translate_btn = gr.Button("🚀 Translate", variant="primary", size="lg")

                with gr.Column(scale=1):
                    gr.Markdown("### Output")

                    output_text = gr.Textbox(
                        label="Vietnamese Translation",
                        lines=6,
                        max_lines=10,
                        interactive=False
                    )

                    info_text = gr.Markdown()

            gr.Markdown("---")
            gr.Markdown("### Example Sentences")

            gr.Examples(
                examples=examples,
                inputs=[input_text, reference_text],
                outputs=[output_text, info_text],
                fn=self.translate,
                cache_examples=False
            )

            with gr.Accordion("Translation History", open=False):
                history_table = gr.Markdown()
                with gr.Row():
                    refresh_btn = gr.Button("Refresh", size="sm")
                    clear_btn = gr.Button("Clear", size="sm", variant="secondary")

                refresh_btn.click(self.get_history, outputs=history_table)
                clear_btn.click(self.clear_history, outputs=history_table)

            gr.Markdown("---")

            # Technical details - pure markdown
            gr.Markdown("""
                ### Technical Details

                **Model Architecture:** BiLSTM Encoder + Multi-Head Attention + LSTM Decoder

                **Features:** Label Smoothing • Layer Normalization • Mixed Precision Training

                **Dataset:** TED Talks (146k sentence pairs) | **Vocab:** 30k EN / 25k VI

                **Training:** Warmup + Cosine Decay LR Schedule | Adam Optimizer
            """)

            gr.Markdown("""
                ---
                <div style="text-align: center; color: #6c757d; font-size: 0.85rem;">
                Made with for educational purposes | Not for production use
                </div>
            """)

            translate_btn.click(
                self.translate,
                inputs=[input_text, reference_text],
                outputs=[output_text, info_text]
            )

            print(f"Launching Gradio at http://0.0.0.0:7860")
            demo.launch(share=share, server_name="0.0.0.0", server_port=7860)

def main():
    """Entry point"""
    app = TranslationApp(
        model_path=f"{Config.ARTIFACT_PATH}/bilstm/checkpoints/best_model.keras",
        tokenizer_path=f"{Config.ARTIFACT_PATH}/tokenizers",
        config=Config
    )
    app.launch(share=True)

if __name__ == '__main__':
    main()