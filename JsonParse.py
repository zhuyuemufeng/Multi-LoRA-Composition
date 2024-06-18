from image_Util import lora_tagger, match_path


class JsonParse:
    def __init__(self, lora_path, prompt):
        loras, reality = match_path(lora_path)
        self.lora_type = reality
        self.lora_list = loras
        tagger = lora_tagger(prompt)
        self.taggers = tagger

    def get_lora_type(self):
        return self.lora_type

    def get_lora_list(self):
        return self.lora_list

    def get_taggers(self):
        return self.taggers