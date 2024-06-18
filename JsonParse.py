from image_Util import lora_tagger, match_path


class JsonParse:
    def __init__(self, lora_type, lora_list, tagger):
        self.lora_type = lora_type
        self.lora_list = lora_list
        self.taggers = lora_tagger(tagger)

    def get_lora_type(self):
        return self.lora_type

    def get_lora_list(self):
        return self.lora_list

    def get_taggers(self):
        return self.taggers