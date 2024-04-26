import re

long_vowels = "áéíóú"
vowels = "aeiou" + long_vowels
diphthongs = ["ai", "ei", "oi", "ui", "au", "eu", "iu", "ou"]
consonants = "bcdfghjklmnpqrstvwxyz"
consonant_clusters = [
    "ly", "qu", "ty", "ny", "hy","hw","hr","th",
    "ll", "pp", "nn", "ss", "tt", "rr", "mm"  # double consonants
]

# 定义音标映射
vowel_map = {
    'a': 'a', 'ä': 'a', 'á': 'aː', 'â': 'a', 'ā': 'a',
    'e': 'ɛ', 'ë': 'ɛ', 'é': 'eː', 'ê': 'ɛ', 'ē': 'ɛ',
    'i': 'i', 'ï': 'i', 'í': 'iː', 'î': 'i', 'ī': 'i',
    'o': 'o', 'ö': 'o', 'ó': 'oː', 'ô': 'o', 'ō': 'o',
    'u': 'u', 'ü': 'u', 'ú': 'uː', 'û': 'u', 'ū': 'u'
}

regular_consonants = { 
    # regular consonants
    'f': 'f', 'h': 'h', 'k': 'k', 'c': 'k', 'l': 'l', 'm': 'm', 'n': 'n',
    'p': 'p', 's': 's', 't': 't', 'v': 'v', 'y': 'j', 'r': 'r', 'g': 'ɡ',
    'þ': 'θ', 'Θ': 'θ', 'θ': 'θ', 'ñ': 'n', 'χ': 'h', 'b': 'b',
    'w': 'v', 'd': 'd', 'x': 'ks','th': 'θ', 
    # not in quenya
    # 'z': 'z', 'j': 'j', 'q': 'k',
}
double_consonants = {    # 双辅音模仿芬兰语的发音模式
    'll': 'll', 'pp': 'pp', 'nn': 'nn', 'ss': 'ss', 'tt': 'tt', 'rr': 'rr', 'mm': 'mm',
    'qu': 'kw', 'ny': 'ɲ', 'ty': 'c','hw': 'ʍ', 'hr': 'ɾ', 'ly': 'ʎ', 'hy': 'ç'
}

vowel_sounds = set(vowel_map.values())
regular_consonants_sounds = set(regular_consonants.values())
double_consonants_sounds = set(double_consonants.values())

def quenya_to_ipa(word):

    word = word.lower()
    
    # Combine all maps
    combined_map = {}
    combined_map.update(regular_consonants)
    combined_map.update(double_consonants)
    combined_map.update(vowel_map)
    
    # Sort keys by length in descending order to handle longer patterns first
    sorted_patterns = sorted(combined_map.keys(), key=len, reverse=True)
    pattern = '|'.join(re.escape(key) for key in sorted_patterns)
    
    # Replace based on sorted patterns
    def replace_func(match):
        return combined_map[match.group(0)]
    
    return re.sub(pattern, replace_func, word)

def split_into_syllables(word):
    syllables = []
    i = 0
    while i < len(word):
        syllable = ''
        while i < len(word) and word[i] not in vowel_sounds:
            syllable += word[i]
            i += 1
        if i < len(word) and word[i] in vowel_sounds:
            if i+1 < len(word) and (word[i] + word[i+1]) in diphthongs:
                syllable += word[i] + word[i+1]
                i += 2
            else:
                syllable += word[i]
                i += 1
        if i < len(word) and i + 1 < len(word) and word[i:i+2] in double_consonants_sounds:
            if syllable:
                syllables.append(syllable)
            syllable = word[i:i+2]
            i += 2
            if i < len(word) and word[i] in vowel_sounds:
                syllable += word[i]
                i += 1
        while i < len(word) and word[i] not in vowel_sounds:
            if i + 1 < len(word) and word[i + 1] in vowel_sounds:
                break
            syllable += word[i]
            i += 1
        if syllable:
            syllables.append(syllable)
    return syllables


def is_syllable_long(syllables, index):
    current_syllable = syllables[index]
    if any(vowel in current_syllable for vowel in ['aː', 'eː', 'iː', 'oː', 'uː']):
        return True
    if any(diph in current_syllable for diph in diphthongs):
        return True
    if index + 1 < len(syllables):
        next_syllable = syllables[index + 1]
        if current_syllable[-1] in regular_consonants_sounds and next_syllable[0] in regular_consonants_sounds:
            return True
        if next_syllable[:2] in ['kw','rr','mm','ss','ll','tt','nn','pp']:
            return True
        if next_syllable[0] in ['c','ʎ', 'ɾ', 'ʍ', 'ç', 'ɲ']:
            return True
    return False

def add_stress_to_vowel(syllable):
    # 这个函数会遍历音节中的字符，寻找第一个元音，并在它前面添加重音符号。
    for index, char in enumerate(syllable):
        if char in vowels:  # 检查字符是否为元音
            return syllable[:index] + '\u02C8' + syllable[index:]
    return syllable  # 如果音节中没有元音，则不添加重音

def apply_stress(word):
    syllables = split_into_syllables(word)
    num_syllables = len(syllables)
    
    # 确定应添加重音的音节索引
    if num_syllables <= 2:
        stressed_index = 0
    else:
        stressed_index = -2  # 默认为倒数第二个音节
        if not is_syllable_long(syllables, -2):
            stressed_index = -3  # 如果倒数第二个音节不是长音节，则选择倒数第三个
    
    # 在选定的音节中添加重音
    syllables[stressed_index] = add_stress_to_vowel(syllables[stressed_index])
    
    return ''.join(syllables)


def process_complete_sentence(text):
    # 定义标点符号
    punctuation_marks = ';:,.!?¡¿—…"«»“”~/。【】、‥،؟“”؛'
    # 使用正则表达式分割单词和标点
    words_and_punct = re.findall(r'\w+|[{}]+'.format(re.escape(punctuation_marks)), text.strip())

    processed_parts = []
    for part in words_and_punct:
        if re.fullmatch(r'\w+', part):
            # 处理单词部分
            g2p = quenya_to_ipa(part)
            stressed_word = apply_stress(g2p)
            processed_parts.append(stressed_word)  # 添加单词
        else:
            # 添加标点
            processed_parts.append(part)

    # 精细控制连接，确保单词后有空格，标点后无空格，然后再处理标点后的空格
    processed_text = ''
    for i, part in enumerate(processed_parts):
        if i + 1 < len(processed_parts) and re.fullmatch(r'\w+', processed_parts[i+1]):
            processed_text += part + ' '  # 如果后一个部分是单词，当前部分后加空格
        else:
            processed_text += part

    return processed_text



