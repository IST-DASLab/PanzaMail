from panza.entities.instruction import EmailInstruction, Instruction
from panza.writer import PanzaWriter
import gradio as gr


class PanzaGUI:
    def __init__(self, writer: PanzaWriter, **kwargs):
        self.writer = writer
        with gr.Blocks() as panza:
            gr.Markdown("# Panza\n")
            inputbox = gr.Textbox(label="Input", placeholder="Enter text and press ENTER")
            outputbox = gr.Textbox(label="Output", placeholder="Generated result from the model")
            inputbox.submit(
                self.get_execute(),
                [inputbox],
                [outputbox],
            )

        panza.queue().launch(server_name="localhost", server_port=5003, share=True)

    def get_execute(self):
        def execute(input):
            instruction: Instruction = EmailInstruction(input)
            stream = self.writer.run(instruction, stream=False)
            # output = ""
            # for chunk in stream:
            #    output += chunk
            # yield stream.end()
            yield stream

        return execute
