import time
import sys
import gradio as gr
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

        self.tokenizer_src = load_tokenizer(f"{tokenizer_path}/tokenizer_src.pkl")
        self.tokenizer_trg = load_tokenizer(f"{tokenizer_path}/tokenizer_trg.pkl")

        self.config = config
        self.bleu_scorer = BLEUScore()
        self.history = []

        self.decoder = BeamSearchDecoder(
            self.model,
            self.tokenizer_src,
            self.tokenizer_trg,
            config['max_length_src'],
            config['max_length_trg'],
            beam_width=config.get('beam_width', 5)
        )

        print("Model loaded!")

    def translate(self, text: str, use_beam_search: bool = False, beam_width: int = 5, reference: str = "") -> tuple:
        """Main translation function"""
        if not text.strip():
            return "", "Please enter text", ""

        start_time = time.time()

        if use_beam_search:
            # Update beam width
            self.decoder.beam_width = beam_width
            translation, candidates = self.decoder.decode_beam_search(
                self.decoder.preprocess(text)
            )
            method = f"Beam Search (k={beam_width})"

            # Format candidates
            candidates_text = "**Alternative translations:**\n\n"
            for i, (trans, score) in enumerate(candidates[:5], 1):
                candidates_text += f"{i}. {trans} (score: {score:.2f})\n"
        else:
            translation = self.decoder.decode_greedy(
                self.decoder.preprocess(text)
            )
            method = "Greedy Decoding"
            candidates_text = ""

        time_taken = time.time() - start_time

        # BLEU score
        bleu_info = ""
        if reference.strip():
            bleu = self.bleu_scorer.compute(reference.lower(), translation.lower())
            bleu_info = f"\nBLEU Score: **{bleu:.2f}**"

        # Info
        info = f"""
            **Translation Info:**
            - Method: {method}
            - Time: {time_taken:.3f}s
            - Input: {len(text.split())} words
            - Output: {len(translation.split())} words{bleu_info}
        """.strip()

        # History
        self.history.append({
            'input': text,
            'output': translation,
            'method': method,
            'time': time_taken
        })

        return translation, info, candidates_text
    def get_history(self) -> str:
        """Get translation history"""
        if not self.history:
            return "No history yet."

        table = "| # | Input | Translation | Time |\n"
        table += "|---|-------|-------------|------|\n"

        for i, item in enumerate(reversed(self.history[-10:]), 1):
            inp = item['input'][:30] + "..." if len(item['input']) > 30 else item['input']
            out = item['output'][:30] + "..." if len(item['output']) > 30 else item['output']
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

        with gr.Blocks(title="Translation", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # English to Vietnamese Translation

            BiLSTM + Multi-Head Attention Model
            """)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Input")
                    input_text = gr.Textbox(
                        label="English Text",
                        placeholder="Enter English text...",
                        lines=5
                    )
                    reference_text = gr.Textbox(
                        label="Reference (Optional - for BLEU)",
                        placeholder="Vietnamese reference...",
                        lines=3
                    )
                    translate_btn = gr.Button("Translate", variant="primary")

                with gr.Column():
                    gr.Markdown("### Output")
                    output_text = gr.Textbox(
                        label="Vietnamese Translation",
                        lines=5
                    )
                    info_text = gr.Markdown()
                    candidates_text = gr.Markdown()

            gr.Markdown("### Examples")
            gr.Examples(
                examples=examples,
                inputs=[input_text, reference_text],
                outputs=[output_text, info_text, candidates_text],
                fn=self.translate
            )

            with gr.Accordion("History", open=False):
                history_table = gr.Markdown()
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                    clear_btn = gr.Button("Clear")

                refresh_btn.click(self.get_history, outputs=history_table)
                clear_btn.click(self.clear_history, outputs=history_table)

            gr.Markdown("""
                ---
                **Model:** BiLSTM + Multi-Head Attention  
                **Features:** Label Smoothing, Layer Normalization, Mixed Precision
            """)

            translate_btn.click(
                self.translate,
                inputs=[input_text, reference_text],
                outputs=[output_text, info_text, candidates_text]
            )

        print(f"Launching Gradio at http://0.0.0.0:7860")
        demo.launch(share=share, server_name="0.0.0.0", server_port=7860)


def main():
    """Entry point"""
    config = Config.to_dict()

    app = TranslationApp(
        model_path=f"{config["model_save_path"]}/bilstm_model.h5",
        tokenizer_path=config["tokenizer_path"],
        config=config
    )
    app.launch(share=True)

if __name__ == '__main__':
    main()
