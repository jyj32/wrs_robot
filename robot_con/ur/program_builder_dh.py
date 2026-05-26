class ProgramBuilder(object):

    def __init__(self):
        self.complete_program = ""

    def load_prog(self, filename):
        self.complete_program = ""
        self._file = open(filename, "r")
        partscript = self._file.read(1024)
        while partscript:
            self.complete_program += partscript
            partscript = self._file.read(1024)

    def get_program_to_run(self):
        if (self.complete_program == ""):
            self.logger.debug("The given script is empty!")
            return ""
        return self.complete_program

    def get_str_from_file(self, filename):
        raw = filename.read_bytes()

        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]

        text = raw.decode("utf-8", errors="replace")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\t", "  ")
        if not text.endswith("\n"):
            text += "\n"
        if "def main" in text and "main()" not in text:
            text += "main()\n"
        # print("=== final script to send ===")
        # # print(text) # 会输出很多代码
        # print("=== end ===")
        return text