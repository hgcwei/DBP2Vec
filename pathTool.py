import re
import os
class pathTool:

    def valid_filename(self,filename):
        invalid_chars = '[\\\/:*?"<>|]'
        replace_char = '-'
        return re.sub(invalid_chars, replace_char, filename)

    def list_all_files(self,dir):
        return os.listdir(dir)

