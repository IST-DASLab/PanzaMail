from panza3.entities.instruction import EmailInstruction, Instruction
from panza3.writer import PanzaWriter


class PanzaCLI:
    def __init__(self, writer: PanzaWriter, **kwargs):
        self.writer = writer
        while True:
            user_input = input("Enter a command: ")
            if user_input == "exit":
                break
            else:
                instruction: Instruction = EmailInstruction(user_input)
                stream = self.writer.run(instruction, stream=True)
                for chunk in stream:
                    print(chunk, end="")
