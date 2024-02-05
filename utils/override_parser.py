import yaml
import json

class OverrideParser:

    def __init__(self, config, overrides):
        self.config = config
        self.overrides = overrides

    def parse_one(self, override):
        override_item = override.split('=')
        assert len(override_item) <= 2
        if len(override_item) == 1:
            key, value = override_item[0], True
        elif len(override_item) == 2:
            key, value = override_item[0], yaml.safe_load(override_item[1])

        override_keys = key.split('.')
        cur_dict = self.config
        max_depth = len(override_keys) - 1
        for idx,k in enumerate(override_keys):
            k, isint = self.safe_parse_int(k)
            if idx == max_depth:
                cur_dict[k] = self.safe_parse_type(value)
            elif isint or (k in cur_dict):
                cur_dict = cur_dict[k]
            else:
                cur_dict[k] = {}
                cur_dict = cur_dict[k]

    def safe_parse_type(self, s):
        try:
            return json.loads(s)
        except:
            return s

    def safe_parse_int(self, s):
        try:
            assert int(s) >= 0
            return int(s), True
        except:
            return s, False

    def parse(self):
        for override in self.overrides:
            self.parse_one(override)
        return self.config


        
