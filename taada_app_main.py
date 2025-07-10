import sys
import spacy
import os
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QApplication, QWidget, QLineEdit, QFileDialog, QGridLayout, QPushButton, QMainWindow,
                             QCheckBox, QLabel, QFrame, QErrorMessage)
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtCore import QObject, QThread, pyqtSignal, QRunnable, Qt, QThreadPool, QProcess
import numpy as np
import en_core_web_sm
import pandas as pd
import random
import subprocess


def resource_path(relative):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative)
    return os.path.join(relative)


if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app 
    # path into variable _MEIPASS'.
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))


def safe_divide(a, b):
    if b != 0:
        return a/b
    else:
        return 0

def remove_nan(list_of_lists):
    return [
        [val for val in sublist if val == val]
        for sublist in list_of_lists
    ]


#====================================================================================

# set up decoding
decoding_df_path = resource_path('decoding_1_dataframe.csv')
decoding = pd.read_csv (decoding_df_path)
decoding_2 = decoding.iloc[:, np.r_[1, 4:56]] #use numpy to call non-consecutive columns
decoding_dic = decoding_2.set_index('words_lower').T.to_dict('list') #make dictionary key and vals (as a list). This makes it easier to call vals in slices
proc = en_core_web_sm.load()


def minicheck(df, df_docs, cands):
    print(cands)
    if "All words lemmatized" in cands:
        df = unlemma_aw(df, df_docs, cands)
    if "All words unlemmatized" in cands:
        df = unlemma_cw(df, df_docs, cands)
    if "Content words lemmatized" in cands:
        df = lemma_aw(df, df_docs, cands)
    if "Content words unlemmatized" in cands:
        df = lemma_cw(df, df_docs, cands)
    return df

def unlemma_aw(df2, df_docs, cands):
    nw_token_text = []
    num_syllables_token_aw_text = []
    num_letters_token_aw_text = []
    num_phonemes_token_aw_text = []
    discrepancy_raw_token_aw_text = []
    discrepancy_ratio_token_aw_text = []
    avg_syllable_length_token_aw_text = []
    num_consonants_characters_token_aw_text = []
    num_vowel_characters_token_aw_text = []
    num_consonants_phonemes_token_aw_text = []
    num_vowel_phonemes_token_aw_text = []
    avg_phonemes_per_character_consonants_token_aw_text = []
    avg_phonemes_per_character_vowels_token_aw_text = []
    avg_phonemes_per_character_all_token_aw_text = []
    prior_prob_cons_token_aw_text = []
    max_prob_cons_token_aw_text = []
    min_prob_cons_token_aw_text = []
    mid_prob_cons_token_aw_text = []
    number_phonemes_cons_token_aw_text = []
    prior_prob_vowel_token_aw_text = []
    max_prob_vowel_token_aw_text = []
    min_prob_vowel_token_aw_text = []
    mid_prob_vowel_token_aw_text = []
    number_phonemes_vowel_token_aw_text = []
    prior_prob_all_token_aw_text = []
    max_prob_all_token_aw_text = []
    mid_prob_all_token_aw_text = []
    min_prob_all_token_aw_text = []
    number_phonemes_all_token_aw_text = []
    Conditional_Probability_Average_token_aw_text = []
    Ortho_N_token_aw_text = []
    Phono_N_token_aw_text = []
    Phono_N_H_token_aw_text = []
    OG_N_token_aw_text = []
    OG_N_H_token_aw_text = []
    Freq_N_token_aw_text = []
    Freq_N_P_token_aw_text = []
    Freq_N_PH_token_aw_text = []
    Freq_N_OG_token_aw_text = []
    Freq_N_OGH_token_aw_text = []
    OLD_token_aw_text = []
    OLDF_token_aw_text = []
    PLD_token_aw_text = []
    PLDF_token_aw_text = []
    subtlexus_log_freq_token_aw_text = []
    subtlexus_log_cd_token_aw_text = []
    coca_maga_cd_token_aw_text = []
    coca_mag_log_freq_token_aw_text = []
    num_rhymes_full_elp_token_aw_text = []
    num_rhymes_1000_coca_token_aw_text = []
    num_rhymes_2500_coca_token_aw_text = []
    num_rhymes_5000_coca_token_aw_text = []
    num_rhymes_10000_coca_token_aw_text = []


    for tokenized_doc in df_docs:
        nw_token = []
        num_syllables_token_aw = []
        num_letters_token_aw = []
        num_phonemes_token_aw = []
        discrepancy_raw_token_aw = []
        discrepancy_ratio_token_aw = []
        avg_syllable_length_token_aw = []
        num_consonants_characters_token_aw = []
        num_vowel_characters_token_aw = []
        num_consonants_phonemes_token_aw = []
        num_vowel_phonemes_token_aw = []
        avg_phonemes_per_character_consonants_token_aw = []
        avg_phonemes_per_character_vowels_token_aw = []
        avg_phonemes_per_character_all_token_aw = []
        prior_prob_cons_token_aw = []
        max_prob_cons_token_aw = []
        min_prob_cons_token_aw = []
        mid_prob_cons_token_aw = []
        number_phonemes_cons_token_aw = []
        prior_prob_vowel_token_aw = []
        max_prob_vowel_token_aw = []
        min_prob_vowel_token_aw = []
        mid_prob_vowel_token_aw = []
        number_phonemes_vowel_token_aw = []
        prior_prob_all_token_aw = []
        max_prob_all_token_aw = []
        mid_prob_all_token_aw = []
        min_prob_all_token_aw = []
        number_phonemes_all_token_aw = []
        Conditional_Probability_Average_token_aw = []
        Ortho_N_token_aw = []
        Phono_N_token_aw = []
        Phono_N_H_token_aw = []
        OG_N_token_aw = []
        OG_N_H_token_aw = []
        Freq_N_token_aw = []
        Freq_N_P_token_aw = []
        Freq_N_PH_token_aw = []
        Freq_N_OG_token_aw = []
        Freq_N_OGH_token_aw = []
        OLD_token_aw = []
        OLDF_token_aw = []
        PLD_token_aw = []
        PLDF_token_aw = []
        subtlexus_log_freq_token_aw = []
        subtlexus_log_cd_token_aw = []
        coca_maga_cd_token_aw = []
        coca_mag_log_freq_token_aw = []
        num_rhymes_full_elp_token_aw = []
        num_rhymes_1000_coca_token_aw = []
        num_rhymes_2500_coca_token_aw = []
        num_rhymes_5000_coca_token_aw = []
        num_rhymes_10000_coca_token_aw = []
        for token in tokenized_doc:
            if token.is_alpha: #remove all non-alpha crap
                nw_token.append(str(token))
                try: 
                #try block encloses the code that might raise an exception (in this case, a KeyError because the val is not there)
                #it executes the code inside the block
                    val = decoding_dic[token.text]
                    #retrieve the value associated with the token
                    num_syllables_token_aw.append(val[0])
                    num_letters_token_aw.append(val[1])
                    num_phonemes_token_aw.append(val[2])
                    discrepancy_raw_token_aw.append(val[3])
                    discrepancy_ratio_token_aw.append(val[4])
                    avg_syllable_length_token_aw.append(val[5])
                    num_consonants_characters_token_aw.append(val[6])
                    num_vowel_characters_token_aw.append(val[7])
                    num_consonants_phonemes_token_aw.append(val[8])
                    num_vowel_phonemes_token_aw.append(val[9])
                    avg_phonemes_per_character_consonants_token_aw.append(val[10])
                    avg_phonemes_per_character_vowels_token_aw.append(val[11])
                    avg_phonemes_per_character_all_token_aw.append(val[12])
                    prior_prob_cons_token_aw.append(val[13])
                    max_prob_cons_token_aw.append(val[14])
                    min_prob_cons_token_aw.append(val[15])
                    mid_prob_cons_token_aw.append(val[16])
                    number_phonemes_cons_token_aw.append(val[17])
                    prior_prob_vowel_token_aw.append(val[18])
                    max_prob_vowel_token_aw.append(val[19])
                    min_prob_vowel_token_aw.append(val[20])
                    mid_prob_vowel_token_aw.append(val[21])
                    number_phonemes_vowel_token_aw.append(val[22])
                    prior_prob_all_token_aw.append(val[23])
                    max_prob_all_token_aw.append(val[24])
                    mid_prob_all_token_aw.append(val[25])
                    min_prob_all_token_aw.append(val[26])
                    number_phonemes_all_token_aw.append(val[27])
                    Conditional_Probability_Average_token_aw.append(val[28])
                    Ortho_N_token_aw.append(val[29])
                    Phono_N_token_aw.append(val[30])
                    Phono_N_H_token_aw.append(val[31])
                    OG_N_token_aw.append(val[32])
                    OG_N_H_token_aw.append(val[33])
                    Freq_N_token_aw.append(val[34])
                    Freq_N_P_token_aw.append(val[35])
                    Freq_N_PH_token_aw.append(val[36])
                    Freq_N_OG_token_aw.append(val[37])
                    Freq_N_OGH_token_aw.append(val[38])
                    OLD_token_aw.append(val[39])
                    OLDF_token_aw.append(val[40])
                    PLD_token_aw.append(val[41])
                    PLDF_token_aw.append(val[42])
                    subtlexus_log_freq_token_aw.append(val[43])
                    subtlexus_log_cd_token_aw.append(val[44])
                    coca_maga_cd_token_aw.append(val[45])
                    coca_mag_log_freq_token_aw.append(val[46])
                    num_rhymes_full_elp_token_aw.append(val[47])
                    num_rhymes_1000_coca_token_aw.append(val[48])
                    num_rhymes_2500_coca_token_aw.append(val[49])
                    num_rhymes_5000_coca_token_aw.append(val[50])
                    num_rhymes_10000_coca_token_aw.append(val[51])
                except:
                #The except block is executed if a KeyError occurs 
                #when the specified key is not found in the dictionary.
                    pass
                    #if a keyerror occurs, just ignore it.
                    
        nw_token_text.append(len(nw_token))
        num_syllables_token_aw_text.append(num_syllables_token_aw)
        num_letters_token_aw_text.append(num_letters_token_aw)
        num_phonemes_token_aw_text.append(num_phonemes_token_aw)
        discrepancy_raw_token_aw_text.append(discrepancy_raw_token_aw)
        discrepancy_ratio_token_aw_text.append(discrepancy_ratio_token_aw)
        avg_syllable_length_token_aw_text.append(avg_syllable_length_token_aw)
        num_consonants_characters_token_aw_text.append(num_consonants_characters_token_aw)
        num_vowel_characters_token_aw_text.append(num_vowel_characters_token_aw)
        num_consonants_phonemes_token_aw_text.append(num_consonants_phonemes_token_aw)
        num_vowel_phonemes_token_aw_text.append(num_vowel_phonemes_token_aw)
        avg_phonemes_per_character_consonants_token_aw_text.append(avg_phonemes_per_character_consonants_token_aw)
        avg_phonemes_per_character_vowels_token_aw_text.append(avg_phonemes_per_character_vowels_token_aw)
        avg_phonemes_per_character_all_token_aw_text.append(avg_phonemes_per_character_all_token_aw)
        prior_prob_cons_token_aw_text.append(prior_prob_cons_token_aw)
        max_prob_cons_token_aw_text.append(max_prob_cons_token_aw)
        min_prob_cons_token_aw_text.append(min_prob_cons_token_aw)
        mid_prob_cons_token_aw_text.append(mid_prob_cons_token_aw)
        number_phonemes_cons_token_aw_text.append(number_phonemes_cons_token_aw)
        prior_prob_vowel_token_aw_text.append(prior_prob_vowel_token_aw)
        max_prob_vowel_token_aw_text.append(max_prob_vowel_token_aw)
        min_prob_vowel_token_aw_text.append(min_prob_vowel_token_aw)
        mid_prob_vowel_token_aw_text.append(mid_prob_vowel_token_aw)
        number_phonemes_vowel_token_aw_text.append(number_phonemes_vowel_token_aw)
        prior_prob_all_token_aw_text.append(prior_prob_all_token_aw)
        max_prob_all_token_aw_text.append(max_prob_all_token_aw)
        mid_prob_all_token_aw_text.append(mid_prob_all_token_aw)
        min_prob_all_token_aw_text.append(min_prob_all_token_aw)
        number_phonemes_all_token_aw_text.append(number_phonemes_all_token_aw)
        Conditional_Probability_Average_token_aw_text.append(Conditional_Probability_Average_token_aw)
        Ortho_N_token_aw_text.append(Ortho_N_token_aw)
        Phono_N_token_aw_text.append(Phono_N_token_aw)
        Phono_N_H_token_aw_text.append(Phono_N_H_token_aw)
        OG_N_token_aw_text.append(OG_N_token_aw)
        OG_N_H_token_aw_text.append(OG_N_H_token_aw)
        Freq_N_token_aw_text.append(Freq_N_token_aw)
        Freq_N_P_token_aw_text.append(Freq_N_P_token_aw)
        Freq_N_PH_token_aw_text.append(Freq_N_PH_token_aw)
        Freq_N_OG_token_aw_text.append(Freq_N_OG_token_aw)
        Freq_N_OGH_token_aw_text.append(Freq_N_OGH_token_aw)
        OLD_token_aw_text.append(OLD_token_aw)
        OLDF_token_aw_text.append(OLDF_token_aw)
        PLD_token_aw_text.append(PLD_token_aw)
        PLDF_token_aw_text.append(PLDF_token_aw)
        subtlexus_log_freq_token_aw_text.append(subtlexus_log_freq_token_aw)
        subtlexus_log_cd_token_aw_text.append(subtlexus_log_cd_token_aw)
        coca_maga_cd_token_aw_text.append(coca_maga_cd_token_aw)
        coca_mag_log_freq_token_aw_text.append(coca_mag_log_freq_token_aw)
        num_rhymes_full_elp_token_aw_text.append(num_rhymes_full_elp_token_aw)
        num_rhymes_1000_coca_token_aw_text.append(num_rhymes_1000_coca_token_aw)
        num_rhymes_2500_coca_token_aw_text.append(num_rhymes_2500_coca_token_aw)
        num_rhymes_5000_coca_token_aw_text.append(num_rhymes_5000_coca_token_aw)
        num_rhymes_10000_coca_token_aw_text.append(num_rhymes_10000_coca_token_aw)

    Conditional_Probability_Average_token_aw_text_no_nan = remove_nan(Conditional_Probability_Average_token_aw_text)
    Ortho_N_token_aw_text_no_nan = remove_nan(Ortho_N_token_aw_text)
    Phono_N_token_aw_text_no_nan = remove_nan(Phono_N_token_aw_text)
    Phono_N_H_token_aw_text_no_nan = remove_nan(Phono_N_H_token_aw_text)
    OG_N_token_aw_text_no_nan = remove_nan(OG_N_token_aw_text)
    OG_N_H_token_aw_text_no_nan = remove_nan(OG_N_H_token_aw_text)
    Freq_N_token_aw_text_no_nan = remove_nan(Freq_N_token_aw_text)
    Freq_N_P_token_aw_text_no_nan = remove_nan(Freq_N_P_token_aw_text)
    Freq_N_PH_token_aw_text_no_nan = remove_nan(Freq_N_PH_token_aw_text)
    Freq_N_OG_token_aw_text_no_nan = remove_nan(Freq_N_OG_token_aw_text)
    Freq_N_OGH_token_aw_text_no_nan = remove_nan(Freq_N_OGH_token_aw_text)
    OLD_token_aw_text_no_nan = remove_nan(OLD_token_aw_text)
    OLDF_token_aw_text_no_nan = remove_nan(OLDF_token_aw_text)
    PLD_token_aw_text_no_nan = remove_nan(PLD_token_aw_text)
    PLDF_token_aw_text_no_nan = remove_nan(PLDF_token_aw_text)
    subtlexus_log_freq_token_aw_text_no_nan = remove_nan(subtlexus_log_freq_token_aw_text)
    subtlexus_log_cd_token_aw_text_no_nan = remove_nan(subtlexus_log_cd_token_aw_text)
    coca_maga_cd_token_aw_text_no_nan = remove_nan(coca_maga_cd_token_aw_text)
    coca_mag_log_freq_token_aw_text_no_nan = remove_nan(coca_mag_log_freq_token_aw_text)
    num_rhymes_full_elp_token_aw_text_no_nan = remove_nan(num_rhymes_full_elp_token_aw_text)
    num_rhymes_1000_coca_token_aw_text_no_nan = remove_nan(num_rhymes_1000_coca_token_aw_text)
    num_rhymes_2500_coca_token_aw_text_no_nan = remove_nan(num_rhymes_2500_coca_token_aw_text)
    num_rhymes_5000_coca_token_aw_text_no_nan = remove_nan(num_rhymes_5000_coca_token_aw_text)
    num_rhymes_10000_coca_token_aw_text_no_nan = remove_nan(num_rhymes_10000_coca_token_aw_text)

    num_syl_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_syllables_token_aw_text] #if it is a sublist, get average, else (if it is not a sublist, empty list, return 0)
    num_let_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_letters_token_aw_text]
    num_phone_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_phonemes_token_aw_text]
    discrepancy_raw_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in discrepancy_raw_token_aw_text]
    discrepancy_ratio_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in discrepancy_ratio_token_aw_text]
    avg_syl_length_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in avg_syllable_length_token_aw_text]
    num_cons_char_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_consonants_characters_token_aw_text]
    num_vowel_char_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_vowel_characters_token_aw_text]
    num_cons_phone_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_consonants_phonemes_token_aw_text]
    num_vowel_phone_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_vowel_phonemes_token_aw_text]
    avg_phone_per_char_cons_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in avg_phonemes_per_character_consonants_token_aw_text]
    avg_phone_per_char_vowel_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in avg_phonemes_per_character_vowels_token_aw_text]
    avg_phone_per_char_all_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in avg_phonemes_per_character_all_token_aw_text]
    prior_prob_cons_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in prior_prob_cons_token_aw_text]
    max_prob_cons_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in max_prob_cons_token_aw_text]
    min_prob_cons_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in min_prob_cons_token_aw_text]
    mid_prob_cons_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in mid_prob_cons_token_aw_text]
    number_phone_cons_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in number_phonemes_cons_token_aw_text]
    prior_prob_vowel_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in prior_prob_vowel_token_aw_text]
    max_prob_vowel_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in max_prob_vowel_token_aw_text]
    min_prob_vowel_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in min_prob_vowel_token_aw_text]
    mid_prob_vowel_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in mid_prob_vowel_token_aw_text]
    number_phone_vowel_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in number_phonemes_vowel_token_aw_text]
    prior_prob_all_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in prior_prob_all_token_aw_text]
    max_prob_all_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in max_prob_all_token_aw_text]
    mid_prob_all_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in mid_prob_all_token_aw_text]
    min_prob_all_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in min_prob_all_token_aw_text]
    number_phone_all_tok_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in number_phonemes_all_token_aw_text]
    Conditional_Probability_Average_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Conditional_Probability_Average_token_aw_text_no_nan]
    Ortho_N_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Ortho_N_token_aw_text_no_nan]
    Phono_N_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Phono_N_token_aw_text_no_nan]
    Phono_N_H_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Phono_N_H_token_aw_text_no_nan]
    OG_N_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in OG_N_token_aw_text_no_nan]
    OG_N_H_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in OG_N_H_token_aw_text_no_nan]
    Freq_N_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Freq_N_token_aw_text_no_nan]
    Freq_N_P_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Freq_N_P_token_aw_text_no_nan]
    Freq_N_PH_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Freq_N_PH_token_aw_text_no_nan]
    Freq_N_OG_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Freq_N_OG_token_aw_text_no_nan]
    Freq_N_OGH_token_aw = [sum(sub_list) / len(sub_list)  if sub_list else 0 for sub_list in Freq_N_OGH_token_aw_text_no_nan]
    OLD_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in OLD_token_aw_text_no_nan]
    OLDF_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in OLDF_token_aw_text_no_nan]
    PLD_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in PLD_token_aw_text_no_nan]
    PLDF_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in PLDF_token_aw_text_no_nan]
    subtlexus_log_freq_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in subtlexus_log_freq_token_aw_text_no_nan]
    subtlexus_log_cd_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in subtlexus_log_cd_token_aw_text_no_nan]
    coca_maga_cd_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in coca_maga_cd_token_aw_text_no_nan]
    coca_mag_log_freq_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in coca_mag_log_freq_token_aw_text_no_nan]
    num_rhymes_full_elp_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_rhymes_full_elp_token_aw_text_no_nan]
    num_rhymes_1000_coca_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_rhymes_1000_coca_token_aw_text_no_nan]
    num_rhymes_2500_coca_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_rhymes_2500_coca_token_aw_text_no_nan]
    num_rhymes_5000_coca_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_rhymes_5000_coca_token_aw_text_no_nan]
    num_rhymes_10000_coca_token_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_rhymes_10000_coca_token_aw_text_no_nan]

    if "Basic counts" in cands:
        df2 = df2.assign(
            num_syl_tok_aw = num_syl_tok_aw,
            num_let_tok_aw = num_let_tok_aw,
            num_phone_tok_aw = num_phone_tok_aw,
            discrepancy_raw_tok_aw = discrepancy_raw_tok_aw,
            discrepancy_ratio_tok_aw = discrepancy_ratio_tok_aw,
            avg_syl_length_tok_aw = avg_syl_length_tok_aw,
            num_cons_char_tok_aw = num_cons_char_tok_aw,
            num_vowel_char_tok_aw = num_vowel_char_tok_aw,
            num_cons_phone_tok_aw = num_cons_phone_tok_aw,
            num_vowel_phone_tok_aw = num_vowel_phone_tok_aw,
            avg_phone_per_char_cons_tok_aw = avg_phone_per_char_cons_tok_aw,
            avg_phone_per_char_vowel_tok_aw = avg_phone_per_char_vowel_tok_aw,
            avg_phone_per_char_all_tok_aw = avg_phone_per_char_all_tok_aw,
            number_phone_vowel_tok_aw = number_phone_vowel_tok_aw,
            number_phone_all_tok_aw = number_phone_all_tok_aw,
            )

    if "Conditional probability" in cands:
        df2 = df2.assign(
            reverse_prior_prob_cons_tok_aw = prior_prob_cons_tok_aw,
            max_prob_cons_tok_aw = max_prob_cons_tok_aw,
            min_prob_cons_tok_aw = min_prob_cons_tok_aw,
            mid_prob_cons_tok_aw = mid_prob_cons_tok_aw,
            number_phone_cons_tok_aw = number_phone_cons_tok_aw,
            reverse_prior_prob_vowel_tok_aw = prior_prob_vowel_tok_aw,
            max_prob_vowel_tok_aw = max_prob_vowel_tok_aw,
            min_prob_vowel_tok_aw = min_prob_vowel_tok_aw,
            mid_prob_vowel_tok_aw = mid_prob_vowel_tok_aw,
            reverse_prior_prob_all_tok_aw = prior_prob_all_tok_aw,
            max_prob_all_tok_aw = max_prob_all_tok_aw,
            mid_prob_all_tok_aw = mid_prob_all_tok_aw,
            min_prob_all_tok_aw = min_prob_all_tok_aw,
            Conditional_Probability_Average_tok_aw = Conditional_Probability_Average_token_aw,
            )

        df2['weight_max_prob_cons_tok_aw'] = df2['max_prob_cons_tok_aw'] / df2['number_phone_cons_tok_aw']
        df2['weight_min_prob_cons_tok_aw'] = df2['min_prob_cons_tok_aw'] / df2['number_phone_cons_tok_aw']
        df2['weight_mid_prob_cons_tok_aw'] = df2['mid_prob_cons_tok_aw'] / df2['number_phone_cons_tok_aw']

        df2['weight_max_prob_vowel_tok_aw'] = df2['max_prob_vowel_tok_aw'] / df2['number_phone_vowel_tok_aw']
        df2['weight_min_prob_vowel_tok_aw'] = df2['min_prob_vowel_tok_aw'] / df2['number_phone_vowel_tok_aw']
        df2['weight_mid_prob_vowel_tok_aw'] = df2['mid_prob_vowel_tok_aw'] / df2['number_phone_vowel_tok_aw']

        df2['weight_max_prob_all_tok_aw'] = df2['max_prob_all_tok_aw'] / df2['number_phone_all_tok_aw']
        df2['weight_mid_prob_all_tok_aw'] = df2['mid_prob_all_tok_aw'] / df2['number_phone_all_tok_aw']
        df2['weight_min_prob_all_tok_aw'] = df2['min_prob_all_tok_aw'] / df2['number_phone_all_tok_aw']

    if "Neighborhood effects" in cands:
        df2 = df2.assign(
            Ortho_N_tok_aw = Ortho_N_token_aw,
            Phono_N_tok_aw = Phono_N_token_aw,
            Phono_N_H_tok_aw = Phono_N_H_token_aw,
            OG_N_tok_aw = OG_N_token_aw,
            OG_N_H_tok_aw = OG_N_H_token_aw,
            Freq_N_tok_aw = Freq_N_token_aw,
            Freq_N_P_tok_aw = Freq_N_P_token_aw,
            Freq_N_PH_tok_aw = Freq_N_PH_token_aw,
            Freq_N_OG_tok_aw = Freq_N_OG_token_aw,
            Freq_N_OGH_tok_aw = Freq_N_OGH_token_aw,
            OLD_tok_aw = OLD_token_aw,
            OLDF_tok_aw = OLDF_token_aw,
            PLD_tok_aw = PLD_token_aw,
            PLDF_tok_aw = PLDF_token_aw,
            )

    if "Word frequency" in cands:
        df2 = df2.assign(
            subtlexus_log_freq_tok_aw = subtlexus_log_freq_token_aw,
            subtlexus_log_cd_tok_aw = subtlexus_log_cd_token_aw,
            coca_maga_cd_tok_aw = coca_maga_cd_token_aw,
            coca_mag_log_freq_tok_aw = coca_mag_log_freq_token_aw,
            )

    if "Rhymes" in cands:
        df2 = df2.assign(
            num_rhymes_full_elp_tok_aw = num_rhymes_full_elp_token_aw,
            num_rhymes_1000_coca_tok_aw = num_rhymes_1000_coca_token_aw,
            num_rhymes_2500_coca_tok_aw = num_rhymes_2500_coca_token_aw,
            num_rhymes_5000_coca_tok_aw = num_rhymes_5000_coca_token_aw,
            num_rhymes_10000_coca_tok_aw = num_rhymes_10000_coca_token_aw,
            )
    
    return df2

def unlemma_cw(df2, df_docs, cands):
    num_syllables_token_cw_text = []
    num_letters_token_cw_text = []
    num_phonemes_token_cw_text = []
    discrepancy_raw_token_cw_text = []
    discrepancy_ratio_token_cw_text = []
    avg_syllable_length_token_cw_text = []
    num_consonants_characters_token_cw_text = []
    num_vowel_characters_token_cw_text = []
    num_consonants_phonemes_token_cw_text = []
    num_vowel_phonemes_token_cw_text = []
    avg_phonemes_per_character_consonants_token_cw_text = []
    avg_phonemes_per_character_vowels_token_cw_text = []
    avg_phonemes_per_character_all_token_cw_text = []
    prior_prob_cons_token_cw_text = []
    max_prob_cons_token_cw_text = []
    min_prob_cons_token_cw_text = []
    mid_prob_cons_token_cw_text = []
    number_phonemes_cons_token_cw_text = []
    prior_prob_vowel_token_cw_text = []
    max_prob_vowel_token_cw_text = []
    min_prob_vowel_token_cw_text = []
    mid_prob_vowel_token_cw_text = []
    number_phonemes_vowel_token_cw_text = []
    prior_prob_all_token_cw_text = []
    max_prob_all_token_cw_text = []
    mid_prob_all_token_cw_text = []
    min_prob_all_token_cw_text = []
    number_phonemes_all_token_cw_text = []
    Conditional_Probability_Average_token_cw_text = []
    Ortho_N_token_cw_text = []
    Phono_N_token_cw_text = []
    Phono_N_H_token_cw_text = []
    OG_N_token_cw_text = []
    OG_N_H_token_cw_text = []
    Freq_N_token_cw_text = []
    Freq_N_P_token_cw_text = []
    Freq_N_PH_token_cw_text = []
    Freq_N_OG_token_cw_text = []
    Freq_N_OGH_token_cw_text = []
    OLD_token_cw_text = []
    OLDF_token_cw_text = []
    PLD_token_cw_text = []
    PLDF_token_cw_text = []
    subtlexus_log_freq_token_cw_text = []
    subtlexus_log_cd_token_cw_text = []
    coca_maga_cd_token_cw_text = []
    coca_mag_log_freq_token_cw_text = []
    num_rhymes_full_elp_token_cw_text = []
    num_rhymes_1000_coca_token_cw_text = []
    num_rhymes_2500_coca_token_cw_text = []
    num_rhymes_5000_coca_token_cw_text = []
    num_rhymes_10000_coca_token_cw_text = []


    for tokenized_doc in df_docs:
        num_syllables_token_cw = []
        num_letters_token_cw = []
        num_phonemes_token_cw = []
        discrepancy_raw_token_cw = []
        discrepancy_ratio_token_cw = []
        avg_syllable_length_token_cw = []
        num_consonants_characters_token_cw = []
        num_vowel_characters_token_cw = []
        num_consonants_phonemes_token_cw = []
        num_vowel_phonemes_token_cw = []
        avg_phonemes_per_character_consonants_token_cw = []
        avg_phonemes_per_character_vowels_token_cw = []
        avg_phonemes_per_character_all_token_cw = []
        prior_prob_cons_token_cw = []
        max_prob_cons_token_cw = []
        min_prob_cons_token_cw = []
        mid_prob_cons_token_cw = []
        number_phonemes_cons_token_cw = []
        prior_prob_vowel_token_cw = []
        max_prob_vowel_token_cw = []
        min_prob_vowel_token_cw = []
        mid_prob_vowel_token_cw = []
        number_phonemes_vowel_token_cw = []
        prior_prob_all_token_cw = []
        max_prob_all_token_cw = []
        mid_prob_all_token_cw = []
        min_prob_all_token_cw = []
        number_phonemes_all_token_cw = []
        Conditional_Probability_Average_token_cw = []
        Ortho_N_token_cw = []
        Phono_N_token_cw = []
        Phono_N_H_token_cw = []
        OG_N_token_cw = []
        OG_N_H_token_cw = []
        Freq_N_token_cw = []
        Freq_N_P_token_cw = []
        Freq_N_PH_token_cw = []
        Freq_N_OG_token_cw = []
        Freq_N_OGH_token_cw = []
        OLD_token_cw = []
        OLDF_token_cw = []
        PLD_token_cw = []
        PLDF_token_cw = []
        subtlexus_log_freq_token_cw = []
        subtlexus_log_cd_token_cw = []
        coca_maga_cd_token_cw = []
        coca_mag_log_freq_token_cw = []
        num_rhymes_full_elp_token_cw = []
        num_rhymes_1000_coca_token_cw = []
        num_rhymes_2500_coca_token_cw = []
        num_rhymes_5000_coca_token_cw = []
        num_rhymes_10000_coca_token_cw = []
        for token in tokenized_doc:
            if not token.is_stop and not token.is_punct:
                try:
                    val = decoding_dic[token.text]
                    num_syllables_token_cw.append(val[0])
                    num_letters_token_cw.append(val[1])
                    num_phonemes_token_cw.append(val[2])
                    discrepancy_raw_token_cw.append(val[3])
                    discrepancy_ratio_token_cw.append(val[4])
                    avg_syllable_length_token_cw.append(val[5])
                    num_consonants_characters_token_cw.append(val[6])
                    num_vowel_characters_token_cw.append(val[7])
                    num_consonants_phonemes_token_cw.append(val[8])
                    num_vowel_phonemes_token_cw.append(val[9])
                    avg_phonemes_per_character_consonants_token_cw.append(val[10])
                    avg_phonemes_per_character_vowels_token_cw.append(val[11])
                    avg_phonemes_per_character_all_token_cw.append(val[12])
                    prior_prob_cons_token_cw.append(val[13])
                    max_prob_cons_token_cw.append(val[14])
                    min_prob_cons_token_cw.append(val[15])
                    mid_prob_cons_token_cw.append(val[16])
                    number_phonemes_cons_token_cw.append(val[17])
                    prior_prob_vowel_token_cw.append(val[18])
                    max_prob_vowel_token_cw.append(val[19])
                    min_prob_vowel_token_cw.append(val[20])
                    mid_prob_vowel_token_cw.append(val[21])
                    number_phonemes_vowel_token_cw.append(val[22])
                    prior_prob_all_token_cw.append(val[23])
                    max_prob_all_token_cw.append(val[24])
                    mid_prob_all_token_cw.append(val[25])
                    min_prob_all_token_cw.append(val[26])
                    number_phonemes_all_token_cw.append(val[27])
                    Conditional_Probability_Average_token_cw.append(val[28])
                    Ortho_N_token_cw.append(val[29])
                    Phono_N_token_cw.append(val[30])
                    Phono_N_H_token_cw.append(val[31])
                    OG_N_token_cw.append(val[32])
                    OG_N_H_token_cw.append(val[33])
                    Freq_N_token_cw.append(val[34])
                    Freq_N_P_token_cw.append(val[35])
                    Freq_N_PH_token_cw.append(val[36])
                    Freq_N_OG_token_cw.append(val[37])
                    Freq_N_OGH_token_cw.append(val[38])
                    OLD_token_cw.append(val[39])
                    OLDF_token_cw.append(val[40])
                    PLD_token_cw.append(val[41])
                    PLDF_token_cw.append(val[42])
                    subtlexus_log_freq_token_cw.append(val[43])
                    subtlexus_log_cd_token_cw.append(val[44])
                    coca_maga_cd_token_cw.append(val[45])
                    coca_mag_log_freq_token_cw.append(val[46])
                    num_rhymes_full_elp_token_cw.append(val[47])
                    num_rhymes_1000_coca_token_cw.append(val[48])
                    num_rhymes_2500_coca_token_cw.append(val[49])
                    num_rhymes_5000_coca_token_cw.append(val[50])
                    num_rhymes_10000_coca_token_cw.append(val[51])
                except:
                    pass

        num_syllables_token_cw_text.append(num_syllables_token_cw)
        num_letters_token_cw_text.append(num_letters_token_cw)
        num_phonemes_token_cw_text.append(num_phonemes_token_cw)
        discrepancy_raw_token_cw_text.append(discrepancy_raw_token_cw)
        discrepancy_ratio_token_cw_text.append(discrepancy_ratio_token_cw)
        avg_syllable_length_token_cw_text.append(avg_syllable_length_token_cw)
        num_consonants_characters_token_cw_text.append(num_consonants_characters_token_cw)
        num_vowel_characters_token_cw_text.append(num_vowel_characters_token_cw)
        num_consonants_phonemes_token_cw_text.append(num_consonants_phonemes_token_cw)
        num_vowel_phonemes_token_cw_text.append(num_vowel_phonemes_token_cw)
        avg_phonemes_per_character_consonants_token_cw_text.append(avg_phonemes_per_character_consonants_token_cw)
        avg_phonemes_per_character_vowels_token_cw_text.append(avg_phonemes_per_character_vowels_token_cw)
        avg_phonemes_per_character_all_token_cw_text.append(avg_phonemes_per_character_all_token_cw)
        prior_prob_cons_token_cw_text.append(prior_prob_cons_token_cw)
        max_prob_cons_token_cw_text.append(max_prob_cons_token_cw)
        min_prob_cons_token_cw_text.append(min_prob_cons_token_cw)
        mid_prob_cons_token_cw_text.append(mid_prob_cons_token_cw)
        number_phonemes_cons_token_cw_text.append(number_phonemes_cons_token_cw)
        prior_prob_vowel_token_cw_text.append(prior_prob_vowel_token_cw)
        max_prob_vowel_token_cw_text.append(max_prob_vowel_token_cw)
        min_prob_vowel_token_cw_text.append(min_prob_vowel_token_cw)
        mid_prob_vowel_token_cw_text.append(mid_prob_vowel_token_cw)
        number_phonemes_vowel_token_cw_text.append(number_phonemes_vowel_token_cw)
        prior_prob_all_token_cw_text.append(prior_prob_all_token_cw)
        max_prob_all_token_cw_text.append(max_prob_all_token_cw)
        mid_prob_all_token_cw_text.append(mid_prob_all_token_cw)
        min_prob_all_token_cw_text.append(min_prob_all_token_cw)
        number_phonemes_all_token_cw_text.append(number_phonemes_all_token_cw)
        Conditional_Probability_Average_token_cw_text.append(Conditional_Probability_Average_token_cw)
        Ortho_N_token_cw_text.append(Ortho_N_token_cw)
        Phono_N_token_cw_text.append(Phono_N_token_cw)
        Phono_N_H_token_cw_text.append(Phono_N_H_token_cw)
        OG_N_token_cw_text.append(OG_N_token_cw)
        OG_N_H_token_cw_text.append(OG_N_H_token_cw)
        Freq_N_token_cw_text.append(Freq_N_token_cw)
        Freq_N_P_token_cw_text.append(Freq_N_P_token_cw)
        Freq_N_PH_token_cw_text.append(Freq_N_PH_token_cw)
        Freq_N_OG_token_cw_text.append(Freq_N_OG_token_cw)
        Freq_N_OGH_token_cw_text.append(Freq_N_OGH_token_cw)
        OLD_token_cw_text.append(OLD_token_cw)
        OLDF_token_cw_text.append(OLDF_token_cw)
        PLD_token_cw_text.append(PLD_token_cw)
        PLDF_token_cw_text.append(PLDF_token_cw)
        subtlexus_log_freq_token_cw_text.append(subtlexus_log_freq_token_cw)
        subtlexus_log_cd_token_cw_text.append(subtlexus_log_cd_token_cw)
        coca_maga_cd_token_cw_text.append(coca_maga_cd_token_cw)
        coca_mag_log_freq_token_cw_text.append(coca_mag_log_freq_token_cw)
        num_rhymes_full_elp_token_cw_text.append(num_rhymes_full_elp_token_cw)
        num_rhymes_1000_coca_token_cw_text.append(num_rhymes_1000_coca_token_cw)
        num_rhymes_2500_coca_token_cw_text.append(num_rhymes_2500_coca_token_cw)
        num_rhymes_5000_coca_token_cw_text.append(num_rhymes_5000_coca_token_cw)
        num_rhymes_10000_coca_token_cw_text.append(num_rhymes_10000_coca_token_cw)


    #remove nan's from the list of lists
    Conditional_Probability_Average_token_cw_text_no_nan = remove_nan(Conditional_Probability_Average_token_cw_text)
    Ortho_N_token_cw_text_no_nan = remove_nan(Ortho_N_token_cw_text)
    Phono_N_token_cw_text_no_nan = remove_nan(Phono_N_token_cw_text)
    Phono_N_H_token_cw_text_no_nan = remove_nan(Phono_N_H_token_cw_text)
    OG_N_token_cw_text_no_nan = remove_nan(OG_N_token_cw_text)
    OG_N_H_token_cw_text_no_nan = remove_nan(OG_N_H_token_cw_text)
    Freq_N_token_cw_text_no_nan = remove_nan(Freq_N_token_cw_text)
    Freq_N_P_token_cw_text_no_nan = remove_nan(Freq_N_P_token_cw_text)
    Freq_N_PH_token_cw_text_no_nan = remove_nan(Freq_N_PH_token_cw_text)
    Freq_N_OG_token_cw_text_no_nan = remove_nan(Freq_N_OG_token_cw_text)
    Freq_N_OGH_token_cw_text_no_nan = remove_nan(Freq_N_OGH_token_cw_text)
    OLD_token_cw_text_no_nan = remove_nan(OLD_token_cw_text)
    OLDF_token_cw_text_no_nan = remove_nan(OLDF_token_cw_text)
    PLD_token_cw_text_no_nan = remove_nan(PLD_token_cw_text)
    PLDF_token_cw_text_no_nan = remove_nan(PLDF_token_cw_text)
    subtlexus_log_freq_token_cw_text_no_nan = remove_nan(subtlexus_log_freq_token_cw_text)
    subtlexus_log_cd_token_cw_text_no_nan = remove_nan(subtlexus_log_cd_token_cw_text)
    coca_maga_cd_token_cw_text_no_nan = remove_nan(coca_maga_cd_token_cw_text)
    coca_mag_log_freq_token_cw_text_no_nan = remove_nan(coca_mag_log_freq_token_cw_text)
    num_rhymes_full_elp_token_cw_text_no_nan = remove_nan(num_rhymes_full_elp_token_cw_text)
    num_rhymes_1000_coca_token_cw_text_no_nan = remove_nan(num_rhymes_1000_coca_token_cw_text)
    num_rhymes_2500_coca_token_cw_text_no_nan = remove_nan(num_rhymes_2500_coca_token_cw_text)
    num_rhymes_5000_coca_token_cw_text_no_nan = remove_nan(num_rhymes_5000_coca_token_cw_text)
    num_rhymes_10000_coca_token_cw_text_no_nan = remove_nan(num_rhymes_10000_coca_token_cw_text)


    #get lists that are average of sublists
    num_syl_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_syllables_token_cw_text] #if it is a sublist, get average, else (if it is not a sublist, empty list, return 0) 
    num_let_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_letters_token_cw_text]
    num_phone_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_phonemes_token_cw_text]
    discrepancy_raw_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in discrepancy_raw_token_cw_text]
    discrepancy_ratio_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in discrepancy_ratio_token_cw_text]
    avg_syl_length_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in avg_syllable_length_token_cw_text]
    num_cons_char_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_consonants_characters_token_cw_text]
    num_vowel_char_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_vowel_characters_token_cw_text]
    num_cons_phone_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_consonants_phonemes_token_cw_text]
    num_vowel_phone_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_vowel_phonemes_token_cw_text]
    avg_phone_per_char_cons_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in avg_phonemes_per_character_consonants_token_cw_text]
    avg_phone_per_char_vowel_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in avg_phonemes_per_character_vowels_token_cw_text]
    avg_phone_per_char_all_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in avg_phonemes_per_character_all_token_cw_text]
    prior_prob_cons_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in prior_prob_cons_token_cw_text]
    max_prob_cons_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in max_prob_cons_token_cw_text]
    min_prob_cons_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in min_prob_cons_token_cw_text]
    mid_prob_cons_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in mid_prob_cons_token_cw_text]
    number_phone_cons_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in number_phonemes_cons_token_cw_text]
    prior_prob_vowel_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in prior_prob_vowel_token_cw_text]
    max_prob_vowel_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in max_prob_vowel_token_cw_text]
    min_prob_vowel_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in min_prob_vowel_token_cw_text]
    mid_prob_vowel_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in mid_prob_vowel_token_cw_text]
    number_phone_vowel_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in number_phonemes_vowel_token_cw_text]
    prior_prob_all_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in prior_prob_all_token_cw_text]
    max_prob_all_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in max_prob_all_token_cw_text]
    mid_prob_all_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in mid_prob_all_token_cw_text]
    min_prob_all_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in min_prob_all_token_cw_text]
    number_phone_all_tok_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in number_phonemes_all_token_cw_text]
    Conditional_Probability_Average_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Conditional_Probability_Average_token_cw_text_no_nan]
    Ortho_N_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Ortho_N_token_cw_text_no_nan]
    Phono_N_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Phono_N_token_cw_text_no_nan]
    Phono_N_H_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Phono_N_H_token_cw_text_no_nan]
    OG_N_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in OG_N_token_cw_text_no_nan]
    OG_N_H_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in OG_N_H_token_cw_text_no_nan]
    Freq_N_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Freq_N_token_cw_text_no_nan]
    Freq_N_P_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Freq_N_P_token_cw_text_no_nan]
    Freq_N_PH_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Freq_N_PH_token_cw_text_no_nan]
    Freq_N_OG_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Freq_N_OG_token_cw_text_no_nan]
    Freq_N_OGH_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Freq_N_OGH_token_cw_text_no_nan]
    OLD_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in OLD_token_cw_text_no_nan]
    OLDF_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in OLDF_token_cw_text_no_nan]
    PLD_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in PLD_token_cw_text_no_nan]
    PLDF_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in PLDF_token_cw_text_no_nan]
    subtlexus_log_freq_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in subtlexus_log_freq_token_cw_text_no_nan]
    subtlexus_log_cd_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in subtlexus_log_cd_token_cw_text_no_nan]
    coca_maga_cd_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in coca_maga_cd_token_cw_text_no_nan]
    coca_mag_log_freq_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in coca_mag_log_freq_token_cw_text_no_nan]
    num_rhymes_full_elp_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_rhymes_full_elp_token_cw_text_no_nan]
    num_rhymes_1000_coca_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_rhymes_1000_coca_token_cw_text_no_nan]
    num_rhymes_2500_coca_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_rhymes_2500_coca_token_cw_text_no_nan]
    num_rhymes_5000_coca_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_rhymes_5000_coca_token_cw_text_no_nan]
    num_rhymes_10000_coca_token_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_rhymes_10000_coca_token_cw_text_no_nan]


    if "Basic counts" in cands:
        df2 = df2.assign(
            num_syl_tok_cw = num_syl_tok_cw,
            num_let_tok_cw = num_let_tok_cw,
            num_phone_tok_cw = num_phone_tok_cw,
            discrepancy_raw_tok_cw = discrepancy_raw_tok_cw,
            discrepancy_ratio_tok_cw = discrepancy_ratio_tok_cw,
            avg_syl_length_tok_cw = avg_syl_length_tok_cw,
            num_cons_char_tok_cw = num_cons_char_tok_cw,
            num_vowel_char_tok_cw = num_vowel_char_tok_cw,
            num_cons_phone_tok_cw = num_cons_phone_tok_cw,
            num_vowel_phone_tok_cw = num_vowel_phone_tok_cw,
            avg_phone_per_char_cons_tok_cw = avg_phone_per_char_cons_tok_cw,
            avg_phone_per_char_vowel_tok_cw = avg_phone_per_char_vowel_tok_cw,
            avg_phone_per_char_all_tok_cw = avg_phone_per_char_all_tok_cw,
            number_phone_vowel_tok_cw = number_phone_vowel_tok_cw,
            number_phone_all_tok_cw = number_phone_all_tok_cw,
            )

    if "Conditional probability" in cands:
        df2 = df2.assign(
            reverse_prior_prob_cons_tok_cw = prior_prob_cons_tok_cw,
            max_prob_cons_tok_cw = max_prob_cons_tok_cw,
            min_prob_cons_tok_cw = min_prob_cons_tok_cw,
            mid_prob_cons_tok_cw = mid_prob_cons_tok_cw,
            number_phone_cons_tok_cw = number_phone_cons_tok_cw,
            reverse_prior_prob_vowel_tok_cw = prior_prob_vowel_tok_cw,
            max_prob_vowel_tok_cw = max_prob_vowel_tok_cw,
            min_prob_vowel_tok_cw = min_prob_vowel_tok_cw,
            mid_prob_vowel_tok_cw = mid_prob_vowel_tok_cw,
            reverse_prior_prob_all_tok_cw = prior_prob_all_tok_cw,
            max_prob_all_tok_cw = max_prob_all_tok_cw,
            mid_prob_all_tok_cw = mid_prob_all_tok_cw,
            min_prob_all_tok_cw = min_prob_all_tok_cw,
            Conditional_Probability_Average_tok_cw = Conditional_Probability_Average_token_cw,
            )

        df2['weight_max_prob_cons_tok_cw'] = df2['max_prob_cons_tok_cw'] / df2['number_phone_cons_tok_cw']
        df2['weight_min_prob_cons_tok_cw'] = df2['min_prob_cons_tok_cw'] / df2['number_phone_cons_tok_cw']
        df2['weight_mid_prob_cons_tok_cw'] = df2['mid_prob_cons_tok_cw'] / df2['number_phone_cons_tok_cw']

        df2['weight_max_prob_vowel_tok_cw'] = df2['max_prob_vowel_tok_cw'] / df2['number_phone_vowel_tok_cw']
        df2['weight_min_prob_vowel_tok_cw'] = df2['min_prob_vowel_tok_cw'] / df2['number_phone_vowel_tok_cw']
        df2['weight_mid_prob_vowel_tok_cw'] = df2['mid_prob_vowel_tok_cw'] / df2['number_phone_vowel_tok_cw']

        df2['weight_max_prob_all_tok_cw'] = df2['max_prob_all_tok_cw'] / df2['number_phone_all_tok_cw']
        df2['weight_mid_prob_all_tok_cw'] = df2['mid_prob_all_tok_cw'] / df2['number_phone_all_tok_cw']
        df2['weight_min_prob_all_tok_cw'] = df2['min_prob_all_tok_cw'] / df2['number_phone_all_tok_cw']

    if "Neighborhood effects" in cands:
        df2 = df2.assign(
            Ortho_N_tok_cw = Ortho_N_token_cw,
            Phono_N_tok_cw = Phono_N_token_cw,
            Phono_N_H_tok_cw = Phono_N_H_token_cw,
            OG_N_tok_cw = OG_N_token_cw,
            OG_N_H_tok_cw = OG_N_H_token_cw,
            Freq_N_tok_cw = Freq_N_token_cw,
            Freq_N_P_tok_cw = Freq_N_P_token_cw,
            Freq_N_PH_tok_cw = Freq_N_PH_token_cw,
            Freq_N_OG_tok_cw = Freq_N_OG_token_cw,
            Freq_N_OGH_tok_cw = Freq_N_OGH_token_cw,
            OLD_tok_cw = OLD_token_cw,
            OLDF_tok_cw = OLDF_token_cw,
            PLD_tok_cw = PLD_token_cw,
            PLDF_tok_cw = PLDF_token_cw,
            )

    if "Word frequency" in cands:
        df2 = df2.assign(
            subtlexus_log_freq_tok_cw = subtlexus_log_freq_token_cw,
            subtlexus_log_cd_tok_cw = subtlexus_log_cd_token_cw,
            coca_maga_cd_tok_cw = coca_maga_cd_token_cw,
            coca_mag_log_freq_tok_cw = coca_mag_log_freq_token_cw,
            )

    if "Rhymes" in cands:
        df2 = df2.assign(
            num_rhymes_full_elp_tok_cw = num_rhymes_full_elp_token_cw,
            num_rhymes_1000_coca_tok_cw = num_rhymes_1000_coca_token_cw,
            num_rhymes_2500_coca_tok_cw = num_rhymes_2500_coca_token_cw,
            num_rhymes_5000_coca_tok_cw = num_rhymes_5000_coca_token_cw,
            num_rhymes_10000_coca_tok_cw = num_rhymes_10000_coca_token_cw,
            )

    return df2

def lemma_aw(df2, df_docs, cands):
    num_syllables_lemma_aw_text = []
    num_letters_lemma_aw_text = []
    num_phonemes_lemma_aw_text = []
    discrepancy_raw_lemma_aw_text = []
    discrepancy_ratio_lemma_aw_text = []
    avg_syllable_length_lemma_aw_text = []
    num_consonants_characters_lemma_aw_text = []
    num_vowel_characters_lemma_aw_text = []
    num_consonants_phonemes_lemma_aw_text = []
    num_vowel_phonemes_lemma_aw_text = []
    avg_phonemes_per_character_consonants_lemma_aw_text = []
    avg_phonemes_per_character_vowels_lemma_aw_text = []
    avg_phonemes_per_character_all_lemma_aw_text = []
    prior_prob_cons_lemma_aw_text = []
    max_prob_cons_lemma_aw_text = []
    min_prob_cons_lemma_aw_text = []
    mid_prob_cons_lemma_aw_text = []
    number_phonemes_cons_lemma_aw_text = []
    prior_prob_vowel_lemma_aw_text = []
    max_prob_vowel_lemma_aw_text = []
    min_prob_vowel_lemma_aw_text = []
    mid_prob_vowel_lemma_aw_text = []
    number_phonemes_vowel_lemma_aw_text = []
    prior_prob_all_lemma_aw_text = []
    max_prob_all_lemma_aw_text = []
    mid_prob_all_lemma_aw_text = []
    min_prob_all_lemma_aw_text = []
    number_phonemes_all_lemma_aw_text = []
    Conditional_Probability_Average_lemma_aw_text = []
    Ortho_N_lemma_aw_text = []
    Phono_N_lemma_aw_text = []
    Phono_N_H_lemma_aw_text = []
    OG_N_lemma_aw_text = []
    OG_N_H_lemma_aw_text = []
    Freq_N_lemma_aw_text = []
    Freq_N_P_lemma_aw_text = []
    Freq_N_PH_lemma_aw_text = []
    Freq_N_OG_lemma_aw_text = []
    Freq_N_OGH_lemma_aw_text = []
    OLD_lemma_aw_text = []
    OLDF_lemma_aw_text = []
    PLD_lemma_aw_text = []
    PLDF_lemma_aw_text = []
    subtlexus_log_freq_lemma_aw_text = []
    subtlexus_log_cd_lemma_aw_text = []
    coca_maga_cd_lemma_aw_text = []
    coca_mag_log_freq_lemma_aw_text = []
    num_rhymes_full_elp_lemma_aw_text = []
    num_rhymes_1000_coca_lemma_aw_text = []
    num_rhymes_2500_coca_lemma_aw_text = []
    num_rhymes_5000_coca_lemma_aw_text = []
    num_rhymes_10000_coca_lemma_aw_text = []

    for tokenized_doc in df_docs:
        num_syllables_lemma_aw = []
        num_letters_lemma_aw = []
        num_phonemes_lemma_aw = []
        discrepancy_raw_lemma_aw = []
        discrepancy_ratio_lemma_aw = []
        avg_syllable_length_lemma_aw = []
        num_consonants_characters_lemma_aw = []
        num_vowel_characters_lemma_aw = []
        num_consonants_phonemes_lemma_aw = []
        num_vowel_phonemes_lemma_aw = []
        avg_phonemes_per_character_consonants_lemma_aw = []
        avg_phonemes_per_character_vowels_lemma_aw = []
        avg_phonemes_per_character_all_lemma_aw = []
        prior_prob_cons_lemma_aw = []
        max_prob_cons_lemma_aw = []
        min_prob_cons_lemma_aw = []
        mid_prob_cons_lemma_aw = []
        number_phonemes_cons_lemma_aw = []
        prior_prob_vowel_lemma_aw = []
        max_prob_vowel_lemma_aw = []
        min_prob_vowel_lemma_aw = []
        mid_prob_vowel_lemma_aw = []
        number_phonemes_vowel_lemma_aw = []
        prior_prob_all_lemma_aw = []
        max_prob_all_lemma_aw = []
        mid_prob_all_lemma_aw = []
        min_prob_all_lemma_aw = []
        number_phonemes_all_lemma_aw = []
        Conditional_Probability_Average_lemma_aw = []
        Ortho_N_lemma_aw = []
        Phono_N_lemma_aw = []
        Phono_N_H_lemma_aw = []
        OG_N_lemma_aw = []
        OG_N_H_lemma_aw = []
        Freq_N_lemma_aw = []
        Freq_N_P_lemma_aw = []
        Freq_N_PH_lemma_aw = []
        Freq_N_OG_lemma_aw = []
        Freq_N_OGH_lemma_aw = []
        OLD_lemma_aw = []
        OLDF_lemma_aw = []
        PLD_lemma_aw = []
        PLDF_lemma_aw = []
        subtlexus_log_freq_lemma_aw = []
        subtlexus_log_cd_lemma_aw = []
        coca_maga_cd_lemma_aw = []
        coca_mag_log_freq_lemma_aw = []
        num_rhymes_full_elp_lemma_aw = []
        num_rhymes_1000_coca_lemma_aw = []
        num_rhymes_2500_coca_lemma_aw = []
        num_rhymes_5000_coca_lemma_aw = []
        num_rhymes_10000_coca_lemma_aw = []
        for token in tokenized_doc:
            if not token.is_punct:
                try:
                    val = decoding_dic[token.lemma_]
                    num_syllables_lemma_aw.append(val[0])
                    num_letters_lemma_aw.append(val[1])
                    num_phonemes_lemma_aw.append(val[2])
                    discrepancy_raw_lemma_aw.append(val[3])
                    discrepancy_ratio_lemma_aw.append(val[4])
                    avg_syllable_length_lemma_aw.append(val[5])
                    num_consonants_characters_lemma_aw.append(val[6])
                    num_vowel_characters_lemma_aw.append(val[7])
                    num_consonants_phonemes_lemma_aw.append(val[8])
                    num_vowel_phonemes_lemma_aw.append(val[9])
                    avg_phonemes_per_character_consonants_lemma_aw.append(val[10])
                    avg_phonemes_per_character_vowels_lemma_aw.append(val[11])
                    avg_phonemes_per_character_all_lemma_aw.append(val[12])
                    prior_prob_cons_lemma_aw.append(val[13])
                    max_prob_cons_lemma_aw.append(val[14])
                    min_prob_cons_lemma_aw.append(val[15])
                    mid_prob_cons_lemma_aw.append(val[16])
                    number_phonemes_cons_lemma_aw.append(val[17])
                    prior_prob_vowel_lemma_aw.append(val[18])
                    max_prob_vowel_lemma_aw.append(val[19])
                    min_prob_vowel_lemma_aw.append(val[20])
                    mid_prob_vowel_lemma_aw.append(val[21])
                    number_phonemes_vowel_lemma_aw.append(val[22])
                    prior_prob_all_lemma_aw.append(val[23])
                    max_prob_all_lemma_aw.append(val[24])
                    mid_prob_all_lemma_aw.append(val[25])
                    min_prob_all_lemma_aw.append(val[26])
                    number_phonemes_all_lemma_aw.append(val[27])
                    Conditional_Probability_Average_lemma_aw.append(val[28])
                    Ortho_N_lemma_aw.append(val[29])
                    Phono_N_lemma_aw.append(val[30])
                    Phono_N_H_lemma_aw.append(val[31])
                    OG_N_lemma_aw.append(val[32])
                    OG_N_H_lemma_aw.append(val[33])
                    Freq_N_lemma_aw.append(val[34])
                    Freq_N_P_lemma_aw.append(val[35])
                    Freq_N_PH_lemma_aw.append(val[36])
                    Freq_N_OG_lemma_aw.append(val[37])
                    Freq_N_OGH_lemma_aw.append(val[38])
                    OLD_lemma_aw.append(val[39])
                    OLDF_lemma_aw.append(val[40])
                    PLD_lemma_aw.append(val[41])
                    PLDF_lemma_aw.append(val[42])
                    subtlexus_log_freq_lemma_aw.append(val[43])
                    subtlexus_log_cd_lemma_aw.append(val[44])
                    coca_maga_cd_lemma_aw.append(val[45])
                    coca_mag_log_freq_lemma_aw.append(val[46])
                    num_rhymes_full_elp_lemma_aw.append(val[47])
                    num_rhymes_1000_coca_lemma_aw.append(val[48])
                    num_rhymes_2500_coca_lemma_aw.append(val[49])
                    num_rhymes_5000_coca_lemma_aw.append(val[50])
                    num_rhymes_10000_coca_lemma_aw.append(val[51])
                except:
                    pass

        num_syllables_lemma_aw_text.append(num_syllables_lemma_aw)
        num_letters_lemma_aw_text.append(num_letters_lemma_aw)
        num_phonemes_lemma_aw_text.append(num_phonemes_lemma_aw)
        discrepancy_raw_lemma_aw_text.append(discrepancy_raw_lemma_aw)
        discrepancy_ratio_lemma_aw_text.append(discrepancy_ratio_lemma_aw)
        avg_syllable_length_lemma_aw_text.append(avg_syllable_length_lemma_aw)
        num_consonants_characters_lemma_aw_text.append(num_consonants_characters_lemma_aw)
        num_vowel_characters_lemma_aw_text.append(num_vowel_characters_lemma_aw)
        num_consonants_phonemes_lemma_aw_text.append(num_consonants_phonemes_lemma_aw)
        num_vowel_phonemes_lemma_aw_text.append(num_vowel_phonemes_lemma_aw)
        avg_phonemes_per_character_consonants_lemma_aw_text.append(avg_phonemes_per_character_consonants_lemma_aw)
        avg_phonemes_per_character_vowels_lemma_aw_text.append(avg_phonemes_per_character_vowels_lemma_aw)
        avg_phonemes_per_character_all_lemma_aw_text.append(avg_phonemes_per_character_all_lemma_aw)
        prior_prob_cons_lemma_aw_text.append(prior_prob_cons_lemma_aw)
        max_prob_cons_lemma_aw_text.append(max_prob_cons_lemma_aw)
        min_prob_cons_lemma_aw_text.append(min_prob_cons_lemma_aw)
        mid_prob_cons_lemma_aw_text.append(mid_prob_cons_lemma_aw)
        number_phonemes_cons_lemma_aw_text.append(number_phonemes_cons_lemma_aw)
        prior_prob_vowel_lemma_aw_text.append(prior_prob_vowel_lemma_aw)
        max_prob_vowel_lemma_aw_text.append(max_prob_vowel_lemma_aw)
        min_prob_vowel_lemma_aw_text.append(min_prob_vowel_lemma_aw)
        mid_prob_vowel_lemma_aw_text.append(mid_prob_vowel_lemma_aw)
        number_phonemes_vowel_lemma_aw_text.append(number_phonemes_vowel_lemma_aw)
        prior_prob_all_lemma_aw_text.append(prior_prob_all_lemma_aw)
        max_prob_all_lemma_aw_text.append(max_prob_all_lemma_aw)
        mid_prob_all_lemma_aw_text.append(mid_prob_all_lemma_aw)
        min_prob_all_lemma_aw_text.append(min_prob_all_lemma_aw)
        number_phonemes_all_lemma_aw_text.append(number_phonemes_all_lemma_aw)
        Conditional_Probability_Average_lemma_aw_text.append(Conditional_Probability_Average_lemma_aw)
        Ortho_N_lemma_aw_text.append(Ortho_N_lemma_aw)
        Phono_N_lemma_aw_text.append(Phono_N_lemma_aw)
        Phono_N_H_lemma_aw_text.append(Phono_N_H_lemma_aw)
        OG_N_lemma_aw_text.append(OG_N_lemma_aw)
        OG_N_H_lemma_aw_text.append(OG_N_H_lemma_aw)
        Freq_N_lemma_aw_text.append(Freq_N_lemma_aw)
        Freq_N_P_lemma_aw_text.append(Freq_N_P_lemma_aw)
        Freq_N_PH_lemma_aw_text.append(Freq_N_PH_lemma_aw)
        Freq_N_OG_lemma_aw_text.append(Freq_N_OG_lemma_aw)
        Freq_N_OGH_lemma_aw_text.append(Freq_N_OGH_lemma_aw)
        OLD_lemma_aw_text.append(OLD_lemma_aw)
        OLDF_lemma_aw_text.append(OLDF_lemma_aw)
        PLD_lemma_aw_text.append(PLD_lemma_aw)
        PLDF_lemma_aw_text.append(PLDF_lemma_aw)
        subtlexus_log_freq_lemma_aw_text.append(subtlexus_log_freq_lemma_aw)
        subtlexus_log_cd_lemma_aw_text.append(subtlexus_log_cd_lemma_aw)
        coca_maga_cd_lemma_aw_text.append(coca_maga_cd_lemma_aw)
        coca_mag_log_freq_lemma_aw_text.append(coca_mag_log_freq_lemma_aw)
        num_rhymes_full_elp_lemma_aw_text.append(num_rhymes_full_elp_lemma_aw)
        num_rhymes_1000_coca_lemma_aw_text.append(num_rhymes_1000_coca_lemma_aw)
        num_rhymes_2500_coca_lemma_aw_text.append(num_rhymes_2500_coca_lemma_aw)
        num_rhymes_5000_coca_lemma_aw_text.append(num_rhymes_5000_coca_lemma_aw)
        num_rhymes_10000_coca_lemma_aw_text.append(num_rhymes_10000_coca_lemma_aw)


    #remove nan's from the list of lists

    Conditional_Probability_Average_lemma_aw_text_no_nan = remove_nan(Conditional_Probability_Average_lemma_aw_text)
    Ortho_N_lemma_aw_text_no_nan = remove_nan(Ortho_N_lemma_aw_text)
    Phono_N_lemma_aw_text_no_nan = remove_nan(Phono_N_lemma_aw_text)
    Phono_N_H_lemma_aw_text_no_nan = remove_nan(Phono_N_H_lemma_aw_text)
    OG_N_lemma_aw_text_no_nan = remove_nan(OG_N_lemma_aw_text)
    OG_N_H_lemma_aw_text_no_nan = remove_nan(OG_N_H_lemma_aw_text)
    Freq_N_lemma_aw_text_no_nan = remove_nan(Freq_N_lemma_aw_text)
    Freq_N_P_lemma_aw_text_no_nan = remove_nan(Freq_N_P_lemma_aw_text)
    Freq_N_PH_lemma_aw_text_no_nan = remove_nan(Freq_N_PH_lemma_aw_text)
    Freq_N_OG_lemma_aw_text_no_nan = remove_nan(Freq_N_OG_lemma_aw_text)
    Freq_N_OGH_lemma_aw_text_no_nan = remove_nan(Freq_N_OGH_lemma_aw_text)
    OLD_lemma_aw_text_no_nan = remove_nan(OLD_lemma_aw_text)
    OLDF_lemma_aw_text_no_nan = remove_nan(OLDF_lemma_aw_text)
    PLD_lemma_aw_text_no_nan = remove_nan(PLD_lemma_aw_text)
    PLDF_lemma_aw_text_no_nan = remove_nan(PLDF_lemma_aw_text)
    subtlexus_log_freq_lemma_aw_text_no_nan = remove_nan(subtlexus_log_freq_lemma_aw_text)
    subtlexus_log_cd_lemma_aw_text_no_nan = remove_nan(subtlexus_log_cd_lemma_aw_text)
    coca_maga_cd_lemma_aw_text_no_nan = remove_nan(coca_maga_cd_lemma_aw_text)
    coca_mag_log_freq_lemma_aw_text_no_nan = remove_nan(coca_mag_log_freq_lemma_aw_text)
    num_rhymes_full_elp_lemma_aw_text_no_nan = remove_nan(num_rhymes_full_elp_lemma_aw_text)
    num_rhymes_1000_coca_lemma_aw_text_no_nan = remove_nan(num_rhymes_1000_coca_lemma_aw_text)
    num_rhymes_2500_coca_lemma_aw_text_no_nan = remove_nan(num_rhymes_2500_coca_lemma_aw_text)
    num_rhymes_5000_coca_lemma_aw_text_no_nan = remove_nan(num_rhymes_5000_coca_lemma_aw_text)
    num_rhymes_10000_coca_lemma_aw_text_no_nan = remove_nan(num_rhymes_10000_coca_lemma_aw_text)


    #get lists that are average of sublists

    num_syl_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_syllables_lemma_aw_text]#if it is a sublist, get average, else (if it is not a sublist, empty list, return 0) 
    num_let_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_letters_lemma_aw_text]
    num_phone_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_phonemes_lemma_aw_text]
    discrepancy_raw_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in discrepancy_raw_lemma_aw_text]
    discrepancy_ratio_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in discrepancy_ratio_lemma_aw_text]
    avg_syl_length_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in avg_syllable_length_lemma_aw_text]
    num_cons_char_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_consonants_characters_lemma_aw_text]
    num_vowel_char_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_vowel_characters_lemma_aw_text]
    num_cons_phone_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_consonants_phonemes_lemma_aw_text]
    num_vowel_phone_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_vowel_phonemes_lemma_aw_text]
    avg_phone_per_char_cons_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in avg_phonemes_per_character_consonants_lemma_aw_text]
    avg_phone_per_char_vowel_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in avg_phonemes_per_character_vowels_lemma_aw_text]
    avg_phone_per_char_all_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in avg_phonemes_per_character_all_lemma_aw_text]
    prior_prob_cons_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in prior_prob_cons_lemma_aw_text]
    max_prob_cons_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in max_prob_cons_lemma_aw_text]
    min_prob_cons_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in min_prob_cons_lemma_aw_text]
    mid_prob_cons_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in mid_prob_cons_lemma_aw_text]
    number_phone_cons_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in number_phonemes_cons_lemma_aw_text]
    prior_prob_vowel_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in prior_prob_vowel_lemma_aw_text]
    max_prob_vowel_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in max_prob_vowel_lemma_aw_text]
    min_prob_vowel_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in min_prob_vowel_lemma_aw_text]
    mid_prob_vowel_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in mid_prob_vowel_lemma_aw_text]
    number_phone_vowel_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in number_phonemes_vowel_lemma_aw_text]
    prior_prob_all_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in prior_prob_all_lemma_aw_text]
    max_prob_all_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in max_prob_all_lemma_aw_text]
    mid_prob_all_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in mid_prob_all_lemma_aw_text]
    min_prob_all_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in min_prob_all_lemma_aw_text]
    number_phone_all_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in number_phonemes_all_lemma_aw_text]
    Conditional_Probability_Average_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Conditional_Probability_Average_lemma_aw_text_no_nan]
    Ortho_N_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Ortho_N_lemma_aw_text_no_nan]
    Phono_N_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Phono_N_lemma_aw_text_no_nan]
    Phono_N_H_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Phono_N_H_lemma_aw_text_no_nan]
    OG_N_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in OG_N_lemma_aw_text_no_nan]
    OG_N_H_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in OG_N_H_lemma_aw_text_no_nan]
    Freq_N_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Freq_N_lemma_aw_text_no_nan]
    Freq_N_P_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Freq_N_P_lemma_aw_text_no_nan]
    Freq_N_PH_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Freq_N_PH_lemma_aw_text_no_nan]
    Freq_N_OG_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Freq_N_OG_lemma_aw_text_no_nan]
    Freq_N_OGH_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Freq_N_OGH_lemma_aw_text_no_nan]
    OLD_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in OLD_lemma_aw_text_no_nan]
    OLDF_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in OLDF_lemma_aw_text_no_nan]
    PLD_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in PLD_lemma_aw_text_no_nan]
    PLDF_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in PLDF_lemma_aw_text_no_nan]
    subtlexus_log_freq_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in subtlexus_log_freq_lemma_aw_text_no_nan]
    subtlexus_log_cd_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in subtlexus_log_cd_lemma_aw_text_no_nan]
    coca_maga_cd_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in coca_maga_cd_lemma_aw_text_no_nan]
    coca_mag_log_freq_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in coca_mag_log_freq_lemma_aw_text_no_nan]
    num_rhymes_full_elp_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_rhymes_full_elp_lemma_aw_text_no_nan]
    num_rhymes_1000_coca_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_rhymes_1000_coca_lemma_aw_text_no_nan]
    num_rhymes_2500_coca_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_rhymes_2500_coca_lemma_aw_text_no_nan]
    num_rhymes_5000_coca_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_rhymes_5000_coca_lemma_aw_text_no_nan]
    num_rhymes_10000_coca_lemma_aw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_rhymes_10000_coca_lemma_aw_text_no_nan]


    if "Basic counts" in cands:
        df2 = df2.assign(
            num_syl_lemma_aw = num_syl_lemma_aw,
            num_let_lemma_aw = num_let_lemma_aw,
            num_phone_lemma_aw = num_phone_lemma_aw,
            discrepancy_raw_lemma_aw = discrepancy_raw_lemma_aw,
            discrepancy_ratio_lemma_aw = discrepancy_ratio_lemma_aw,
            avg_syl_length_lemma_aw = avg_syl_length_lemma_aw,
            num_cons_char_lemma_aw = num_cons_char_lemma_aw,
            num_vowel_char_lemma_aw = num_vowel_char_lemma_aw,
            num_cons_phone_lemma_aw = num_cons_phone_lemma_aw,
            num_vowel_phone_lemma_aw = num_vowel_phone_lemma_aw,
            avg_phone_per_char_cons_lemma_aw = avg_phone_per_char_cons_lemma_aw,
            avg_phone_per_char_vowel_lemma_aw = avg_phone_per_char_vowel_lemma_aw,
            avg_phone_per_char_all_lemma_aw = avg_phone_per_char_all_lemma_aw,
            number_phone_vowel_lemma_aw = number_phone_vowel_lemma_aw,
            number_phone_all_lemma_aw = number_phone_all_lemma_aw,
            )

    if "Conditional probability" in cands:
        df2 = df2.assign(
            reverse_prior_prob_cons_lemma_aw = prior_prob_cons_lemma_aw,
            max_prob_cons_lemma_aw = max_prob_cons_lemma_aw,
            min_prob_cons_lemma_aw = min_prob_cons_lemma_aw,
            mid_prob_cons_lemma_aw = mid_prob_cons_lemma_aw,
            number_phone_cons_lemma_aw = number_phone_cons_lemma_aw,
            reverse_prior_prob_vowel_lemma_aw = prior_prob_vowel_lemma_aw,
            max_prob_vowel_lemma_aw = max_prob_vowel_lemma_aw,
            min_prob_vowel_lemma_aw = min_prob_vowel_lemma_aw,
            mid_prob_vowel_lemma_aw = mid_prob_vowel_lemma_aw,
            reverse_prior_prob_all_lemma_aw = prior_prob_all_lemma_aw,
            max_prob_all_lemma_aw = max_prob_all_lemma_aw,
            mid_prob_all_lemma_aw = mid_prob_all_lemma_aw,
            min_prob_all_lemma_aw = min_prob_all_lemma_aw,
            Conditional_Probability_Average_lemma_aw = Conditional_Probability_Average_lemma_aw,
            )

        df2['weight_max_prob_cons_lemma_aw'] = df2['max_prob_cons_lemma_aw'] / df2['number_phone_cons_lemma_aw']
        df2['weight_min_prob_cons_lemma_aw'] = df2['min_prob_cons_lemma_aw'] / df2['number_phone_cons_lemma_aw']
        df2['weight_mid_prob_cons_lemma_aw'] = df2['mid_prob_cons_lemma_aw'] / df2['number_phone_cons_lemma_aw']

        df2['weight_max_prob_vowel_lemma_aw'] = df2['max_prob_vowel_lemma_aw'] / df2['number_phone_vowel_lemma_aw']
        df2['weight_min_prob_vowel_lemma_aw'] = df2['min_prob_vowel_lemma_aw'] / df2['number_phone_vowel_lemma_aw']
        df2['weight_mid_prob_vowel_lemma_aw'] = df2['mid_prob_vowel_lemma_aw'] / df2['number_phone_vowel_lemma_aw']

        df2['weight_max_prob_all_lemma_aw'] = df2['max_prob_all_lemma_aw'] / df2['number_phone_all_lemma_aw']
        df2['weight_mid_prob_all_lemma_aw'] = df2['mid_prob_all_lemma_aw'] / df2['number_phone_all_lemma_aw']
        df2['weight_min_prob_all_lemma_aw'] = df2['min_prob_all_lemma_aw'] / df2['number_phone_all_lemma_aw']

    if "Neighborhood effects" in cands:
        df2 = df2.assign(
            Ortho_N_lemma_aw = Ortho_N_lemma_aw,
            Phono_N_lemma_aw = Phono_N_lemma_aw,
            Phono_N_H_lemma_aw = Phono_N_H_lemma_aw,
            OG_N_lemma_aw = OG_N_lemma_aw,
            OG_N_H_lemma_aw = OG_N_H_lemma_aw,
            Freq_N_lemma_aw = Freq_N_lemma_aw,
            Freq_N_P_lemma_aw = Freq_N_P_lemma_aw,
            Freq_N_PH_lemma_aw = Freq_N_PH_lemma_aw,
            Freq_N_OG_lemma_aw = Freq_N_OG_lemma_aw,
            Freq_N_OGH_lemma_aw = Freq_N_OGH_lemma_aw,
            OLD_lemma_aw = OLD_lemma_aw,
            OLDF_lemma_aw = OLDF_lemma_aw,
            PLD_lemma_aw = PLD_lemma_aw,
            PLDF_lemma_aw = PLDF_lemma_aw,
            )

    if "Word frequency" in cands:
        df2 = df2.assign(
            subtlexus_log_freq_lemma_aw = subtlexus_log_freq_lemma_aw,
            subtlexus_log_cd_lemma_aw = subtlexus_log_cd_lemma_aw,
            coca_maga_cd_lemma_aw = coca_maga_cd_lemma_aw,
            coca_mag_log_freq_lemma_aw = coca_mag_log_freq_lemma_aw,
            )

    if "Rhymes" in cands:
        df2 = df2.assign(
            num_rhymes_full_elp_lemma_aw = num_rhymes_full_elp_lemma_aw,
            num_rhymes_1000_coca_lemma_aw = num_rhymes_1000_coca_lemma_aw,
            num_rhymes_2500_coca_lemma_aw = num_rhymes_2500_coca_lemma_aw,
            num_rhymes_5000_coca_lemma_aw = num_rhymes_5000_coca_lemma_aw,
            num_rhymes_10000_coca_lemma_aw = num_rhymes_10000_coca_lemma_aw,
            )

    return df2

def lemma_cw(df2, df_docs, cands):
    num_syllables_lemma_cw_text = []
    num_letters_lemma_cw_text = []
    num_phonemes_lemma_cw_text = []
    discrepancy_raw_lemma_cw_text = []
    discrepancy_ratio_lemma_cw_text = []
    avg_syllable_length_lemma_cw_text = []
    num_consonants_characters_lemma_cw_text = []
    num_vowel_characters_lemma_cw_text = []
    num_consonants_phonemes_lemma_cw_text = []
    num_vowel_phonemes_lemma_cw_text = []
    avg_phonemes_per_character_consonants_lemma_cw_text = []
    avg_phonemes_per_character_vowels_lemma_cw_text = []
    avg_phonemes_per_character_all_lemma_cw_text = []
    prior_prob_cons_lemma_cw_text = []
    max_prob_cons_lemma_cw_text = []
    min_prob_cons_lemma_cw_text = []
    mid_prob_cons_lemma_cw_text = []
    number_phonemes_cons_lemma_cw_text = []
    prior_prob_vowel_lemma_cw_text = []
    max_prob_vowel_lemma_cw_text = []
    min_prob_vowel_lemma_cw_text = []
    mid_prob_vowel_lemma_cw_text = []
    number_phonemes_vowel_lemma_cw_text = []
    prior_prob_all_lemma_cw_text = []
    max_prob_all_lemma_cw_text = []
    mid_prob_all_lemma_cw_text = []
    min_prob_all_lemma_cw_text = []
    number_phonemes_all_lemma_cw_text = []
    Conditional_Probability_Average_lemma_cw_text = []
    Ortho_N_lemma_cw_text = []
    Phono_N_lemma_cw_text = []
    Phono_N_H_lemma_cw_text = []
    OG_N_lemma_cw_text = []
    OG_N_H_lemma_cw_text = []
    Freq_N_lemma_cw_text = []
    Freq_N_P_lemma_cw_text = []
    Freq_N_PH_lemma_cw_text = []
    Freq_N_OG_lemma_cw_text = []
    Freq_N_OGH_lemma_cw_text = []
    OLD_lemma_cw_text = []
    OLDF_lemma_cw_text = []
    PLD_lemma_cw_text = []
    PLDF_lemma_cw_text = []
    subtlexus_log_freq_lemma_cw_text = []
    subtlexus_log_cd_lemma_cw_text = []
    coca_maga_cd_lemma_cw_text = []
    coca_mag_log_freq_lemma_cw_text = []
    num_rhymes_full_elp_lemma_cw_text = []
    num_rhymes_1000_coca_lemma_cw_text = []
    num_rhymes_2500_coca_lemma_cw_text = []
    num_rhymes_5000_coca_lemma_cw_text = []
    num_rhymes_10000_coca_lemma_cw_text = []

    for tokenized_doc in df_docs:
        num_syllables_lemma_cw = []
        num_letters_lemma_cw = []
        num_phonemes_lemma_cw = []
        discrepancy_raw_lemma_cw = []
        discrepancy_ratio_lemma_cw = []
        avg_syllable_length_lemma_cw = []
        num_consonants_characters_lemma_cw = []
        num_vowel_characters_lemma_cw = []
        num_consonants_phonemes_lemma_cw = []
        num_vowel_phonemes_lemma_cw = []
        avg_phonemes_per_character_consonants_lemma_cw = []
        avg_phonemes_per_character_vowels_lemma_cw = []
        avg_phonemes_per_character_all_lemma_cw = []
        prior_prob_cons_lemma_cw = []
        max_prob_cons_lemma_cw = []
        min_prob_cons_lemma_cw = []
        mid_prob_cons_lemma_cw = []
        number_phonemes_cons_lemma_cw = []
        prior_prob_vowel_lemma_cw = []
        max_prob_vowel_lemma_cw = []
        min_prob_vowel_lemma_cw = []
        mid_prob_vowel_lemma_cw = []
        number_phonemes_vowel_lemma_cw = []
        prior_prob_all_lemma_cw = []
        max_prob_all_lemma_cw = []
        mid_prob_all_lemma_cw = []
        min_prob_all_lemma_cw = []
        number_phonemes_all_lemma_cw = []
        Conditional_Probability_Average_lemma_cw = []
        Ortho_N_lemma_cw = []
        Phono_N_lemma_cw = []
        Phono_N_H_lemma_cw = []
        OG_N_lemma_cw = []
        OG_N_H_lemma_cw = []
        Freq_N_lemma_cw = []
        Freq_N_P_lemma_cw = []
        Freq_N_PH_lemma_cw = []
        Freq_N_OG_lemma_cw = []
        Freq_N_OGH_lemma_cw = []
        OLD_lemma_cw = []
        OLDF_lemma_cw = []
        PLD_lemma_cw = []
        PLDF_lemma_cw = []
        subtlexus_log_freq_lemma_cw = []
        subtlexus_log_cd_lemma_cw = []
        coca_maga_cd_lemma_cw = []
        coca_mag_log_freq_lemma_cw = []
        num_rhymes_full_elp_lemma_cw = []
        num_rhymes_1000_coca_lemma_cw = []
        num_rhymes_2500_coca_lemma_cw = []
        num_rhymes_5000_coca_lemma_cw = []
        num_rhymes_10000_coca_lemma_cw = []
        for token in tokenized_doc:
            if not token.is_stop and not token.is_punct:
                try:
                    val = decoding_dic[token.lemma_]
                    num_syllables_lemma_cw.append(val[0])
                    num_letters_lemma_cw.append(val[1])
                    num_phonemes_lemma_cw.append(val[2])
                    discrepancy_raw_lemma_cw.append(val[3])
                    discrepancy_ratio_lemma_cw.append(val[4])
                    avg_syllable_length_lemma_cw.append(val[5])
                    num_consonants_characters_lemma_cw.append(val[6])
                    num_vowel_characters_lemma_cw.append(val[7])
                    num_consonants_phonemes_lemma_cw.append(val[8])
                    num_vowel_phonemes_lemma_cw.append(val[9])
                    avg_phonemes_per_character_consonants_lemma_cw.append(val[10])
                    avg_phonemes_per_character_vowels_lemma_cw.append(val[11])
                    avg_phonemes_per_character_all_lemma_cw.append(val[12])
                    prior_prob_cons_lemma_cw.append(val[13])
                    max_prob_cons_lemma_cw.append(val[14])
                    min_prob_cons_lemma_cw.append(val[15])
                    mid_prob_cons_lemma_cw.append(val[16])
                    number_phonemes_cons_lemma_cw.append(val[17])
                    prior_prob_vowel_lemma_cw.append(val[18])
                    max_prob_vowel_lemma_cw.append(val[19])
                    min_prob_vowel_lemma_cw.append(val[20])
                    mid_prob_vowel_lemma_cw.append(val[21])
                    number_phonemes_vowel_lemma_cw.append(val[22])
                    prior_prob_all_lemma_cw.append(val[23])
                    max_prob_all_lemma_cw.append(val[24])
                    mid_prob_all_lemma_cw.append(val[25])
                    min_prob_all_lemma_cw.append(val[26])
                    number_phonemes_all_lemma_cw.append(val[27])
                    Conditional_Probability_Average_lemma_cw.append(val[28])
                    Ortho_N_lemma_cw.append(val[29])
                    Phono_N_lemma_cw.append(val[30])
                    Phono_N_H_lemma_cw.append(val[31])
                    OG_N_lemma_cw.append(val[32])
                    OG_N_H_lemma_cw.append(val[33])
                    Freq_N_lemma_cw.append(val[34])
                    Freq_N_P_lemma_cw.append(val[35])
                    Freq_N_PH_lemma_cw.append(val[36])
                    Freq_N_OG_lemma_cw.append(val[37])
                    Freq_N_OGH_lemma_cw.append(val[38])
                    OLD_lemma_cw.append(val[39])
                    OLDF_lemma_cw.append(val[40])
                    PLD_lemma_cw.append(val[41])
                    PLDF_lemma_cw.append(val[42])
                    subtlexus_log_freq_lemma_cw.append(val[43])
                    subtlexus_log_cd_lemma_cw.append(val[44])
                    coca_maga_cd_lemma_cw.append(val[45])
                    coca_mag_log_freq_lemma_cw.append(val[46])
                    num_rhymes_full_elp_lemma_cw.append(val[47])
                    num_rhymes_1000_coca_lemma_cw.append(val[48])
                    num_rhymes_2500_coca_lemma_cw.append(val[49])
                    num_rhymes_5000_coca_lemma_cw.append(val[50])
                    num_rhymes_10000_coca_lemma_cw.append(val[51])
                except:
                    pass

        num_syllables_lemma_cw_text.append(num_syllables_lemma_cw)
        num_letters_lemma_cw_text.append(num_letters_lemma_cw)
        num_phonemes_lemma_cw_text.append(num_phonemes_lemma_cw)
        discrepancy_raw_lemma_cw_text.append(discrepancy_raw_lemma_cw)
        discrepancy_ratio_lemma_cw_text.append(discrepancy_ratio_lemma_cw)
        avg_syllable_length_lemma_cw_text.append(avg_syllable_length_lemma_cw)
        num_consonants_characters_lemma_cw_text.append(num_consonants_characters_lemma_cw)
        num_vowel_characters_lemma_cw_text.append(num_vowel_characters_lemma_cw)
        num_consonants_phonemes_lemma_cw_text.append(num_consonants_phonemes_lemma_cw)
        num_vowel_phonemes_lemma_cw_text.append(num_vowel_phonemes_lemma_cw)
        avg_phonemes_per_character_consonants_lemma_cw_text.append(avg_phonemes_per_character_consonants_lemma_cw)
        avg_phonemes_per_character_vowels_lemma_cw_text.append(avg_phonemes_per_character_vowels_lemma_cw)
        avg_phonemes_per_character_all_lemma_cw_text.append(avg_phonemes_per_character_all_lemma_cw)
        prior_prob_cons_lemma_cw_text.append(prior_prob_cons_lemma_cw)
        max_prob_cons_lemma_cw_text.append(max_prob_cons_lemma_cw)
        min_prob_cons_lemma_cw_text.append(min_prob_cons_lemma_cw)
        mid_prob_cons_lemma_cw_text.append(mid_prob_cons_lemma_cw)
        number_phonemes_cons_lemma_cw_text.append(number_phonemes_cons_lemma_cw)
        prior_prob_vowel_lemma_cw_text.append(prior_prob_vowel_lemma_cw)
        max_prob_vowel_lemma_cw_text.append(max_prob_vowel_lemma_cw)
        min_prob_vowel_lemma_cw_text.append(min_prob_vowel_lemma_cw)
        mid_prob_vowel_lemma_cw_text.append(mid_prob_vowel_lemma_cw)
        number_phonemes_vowel_lemma_cw_text.append(number_phonemes_vowel_lemma_cw)
        prior_prob_all_lemma_cw_text.append(prior_prob_all_lemma_cw)
        max_prob_all_lemma_cw_text.append(max_prob_all_lemma_cw)
        mid_prob_all_lemma_cw_text.append(mid_prob_all_lemma_cw)
        min_prob_all_lemma_cw_text.append(min_prob_all_lemma_cw)
        number_phonemes_all_lemma_cw_text.append(number_phonemes_all_lemma_cw)
        Conditional_Probability_Average_lemma_cw_text.append(Conditional_Probability_Average_lemma_cw)
        Ortho_N_lemma_cw_text.append(Ortho_N_lemma_cw)
        Phono_N_lemma_cw_text.append(Phono_N_lemma_cw)
        Phono_N_H_lemma_cw_text.append(Phono_N_H_lemma_cw)
        OG_N_lemma_cw_text.append(OG_N_lemma_cw)
        OG_N_H_lemma_cw_text.append(OG_N_H_lemma_cw)
        Freq_N_lemma_cw_text.append(Freq_N_lemma_cw)
        Freq_N_P_lemma_cw_text.append(Freq_N_P_lemma_cw)
        Freq_N_PH_lemma_cw_text.append(Freq_N_PH_lemma_cw)
        Freq_N_OG_lemma_cw_text.append(Freq_N_OG_lemma_cw)
        Freq_N_OGH_lemma_cw_text.append(Freq_N_OGH_lemma_cw)
        OLD_lemma_cw_text.append(OLD_lemma_cw)
        OLDF_lemma_cw_text.append(OLDF_lemma_cw)
        PLD_lemma_cw_text.append(PLD_lemma_cw)
        PLDF_lemma_cw_text.append(PLDF_lemma_cw)
        subtlexus_log_freq_lemma_cw_text.append(subtlexus_log_freq_lemma_cw)
        subtlexus_log_cd_lemma_cw_text.append(subtlexus_log_cd_lemma_cw)
        coca_maga_cd_lemma_cw_text.append(coca_maga_cd_lemma_cw)
        coca_mag_log_freq_lemma_cw_text.append(coca_mag_log_freq_lemma_cw)
        num_rhymes_full_elp_lemma_cw_text.append(num_rhymes_full_elp_lemma_cw)
        num_rhymes_1000_coca_lemma_cw_text.append(num_rhymes_1000_coca_lemma_cw)
        num_rhymes_2500_coca_lemma_cw_text.append(num_rhymes_2500_coca_lemma_cw)
        num_rhymes_5000_coca_lemma_cw_text.append(num_rhymes_5000_coca_lemma_cw)
        num_rhymes_10000_coca_lemma_cw_text.append(num_rhymes_10000_coca_lemma_cw)

    #remove nan's from the list of lists

    Conditional_Probability_Average_lemma_cw_text_no_nan = remove_nan(Conditional_Probability_Average_lemma_cw_text)
    Ortho_N_lemma_cw_text_no_nan = remove_nan(Ortho_N_lemma_cw_text)
    Phono_N_lemma_cw_text_no_nan = remove_nan(Phono_N_lemma_cw_text)
    Phono_N_H_lemma_cw_text_no_nan = remove_nan(Phono_N_H_lemma_cw_text)
    OG_N_lemma_cw_text_no_nan = remove_nan(OG_N_lemma_cw_text)
    OG_N_H_lemma_cw_text_no_nan = remove_nan(OG_N_H_lemma_cw_text)
    Freq_N_lemma_cw_text_no_nan = remove_nan(Freq_N_lemma_cw_text)
    Freq_N_P_lemma_cw_text_no_nan = remove_nan(Freq_N_P_lemma_cw_text)
    Freq_N_PH_lemma_cw_text_no_nan = remove_nan(Freq_N_PH_lemma_cw_text)
    Freq_N_OG_lemma_cw_text_no_nan = remove_nan(Freq_N_OG_lemma_cw_text)
    Freq_N_OGH_lemma_cw_text_no_nan = remove_nan(Freq_N_OGH_lemma_cw_text)
    OLD_lemma_cw_text_no_nan = remove_nan(OLD_lemma_cw_text)
    OLDF_lemma_cw_text_no_nan = remove_nan(OLDF_lemma_cw_text)
    PLD_lemma_cw_text_no_nan = remove_nan(PLD_lemma_cw_text)
    PLDF_lemma_cw_text_no_nan = remove_nan(PLDF_lemma_cw_text)
    subtlexus_log_freq_lemma_cw_text_no_nan = remove_nan(subtlexus_log_freq_lemma_cw_text)
    subtlexus_log_cd_lemma_cw_text_no_nan = remove_nan(subtlexus_log_cd_lemma_cw_text)
    coca_maga_cd_lemma_cw_text_no_nan = remove_nan(coca_maga_cd_lemma_cw_text)
    coca_mag_log_freq_lemma_cw_text_no_nan = remove_nan(coca_mag_log_freq_lemma_cw_text)
    num_rhymes_full_elp_lemma_cw_text_no_nan = remove_nan(num_rhymes_full_elp_lemma_cw_text)
    num_rhymes_1000_coca_lemma_cw_text_no_nan = remove_nan(num_rhymes_1000_coca_lemma_cw_text)
    num_rhymes_2500_coca_lemma_cw_text_no_nan = remove_nan(num_rhymes_2500_coca_lemma_cw_text)
    num_rhymes_5000_coca_lemma_cw_text_no_nan = remove_nan(num_rhymes_5000_coca_lemma_cw_text)
    num_rhymes_10000_coca_lemma_cw_text_no_nan = remove_nan(num_rhymes_10000_coca_lemma_cw_text)


    #get lists that are average of sublists

    num_syl_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_syllables_lemma_cw_text]#if it is a sublist, get average, else (if it is not a sublist, empty list, return 0) 
    num_let_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_letters_lemma_cw_text]
    num_phone_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_phonemes_lemma_cw_text]
    discrepancy_raw_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in discrepancy_raw_lemma_cw_text]
    discrepancy_ratio_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in discrepancy_ratio_lemma_cw_text]
    avg_syl_length_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in avg_syllable_length_lemma_cw_text]
    num_cons_char_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_consonants_characters_lemma_cw_text]
    num_vowel_char_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_vowel_characters_lemma_cw_text]
    num_cons_phone_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_consonants_phonemes_lemma_cw_text]
    num_vowel_phone_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_vowel_phonemes_lemma_cw_text]
    avg_phone_per_char_cons_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in avg_phonemes_per_character_consonants_lemma_cw_text]
    avg_phone_per_char_vowel_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in avg_phonemes_per_character_vowels_lemma_cw_text]
    avg_phone_per_char_all_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in avg_phonemes_per_character_all_lemma_cw_text]
    prior_prob_cons_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in prior_prob_cons_lemma_cw_text]
    max_prob_cons_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in max_prob_cons_lemma_cw_text]
    min_prob_cons_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in min_prob_cons_lemma_cw_text]
    mid_prob_cons_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in mid_prob_cons_lemma_cw_text]
    number_phone_cons_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in number_phonemes_cons_lemma_cw_text]
    prior_prob_vowel_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in prior_prob_vowel_lemma_cw_text]
    max_prob_vowel_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in max_prob_vowel_lemma_cw_text]
    min_prob_vowel_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in min_prob_vowel_lemma_cw_text]
    mid_prob_vowel_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in mid_prob_vowel_lemma_cw_text]
    number_phone_vowel_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in number_phonemes_vowel_lemma_cw_text]
    prior_prob_all_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in prior_prob_all_lemma_cw_text]
    max_prob_all_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in max_prob_all_lemma_cw_text]
    mid_prob_all_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in mid_prob_all_lemma_cw_text]
    min_prob_all_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in min_prob_all_lemma_cw_text]
    number_phone_all_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in number_phonemes_all_lemma_cw_text]
    Conditional_Probability_Average_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Conditional_Probability_Average_lemma_cw_text_no_nan]
    Ortho_N_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Ortho_N_lemma_cw_text_no_nan]
    Phono_N_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Phono_N_lemma_cw_text_no_nan]
    Phono_N_H_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Phono_N_H_lemma_cw_text_no_nan]
    OG_N_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in OG_N_lemma_cw_text_no_nan]
    OG_N_H_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in OG_N_H_lemma_cw_text_no_nan]
    Freq_N_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Freq_N_lemma_cw_text_no_nan]
    Freq_N_P_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Freq_N_P_lemma_cw_text_no_nan]
    Freq_N_PH_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Freq_N_PH_lemma_cw_text_no_nan]
    Freq_N_OG_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Freq_N_OG_lemma_cw_text_no_nan]
    Freq_N_OGH_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in Freq_N_OGH_lemma_cw_text_no_nan]
    OLD_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in OLD_lemma_cw_text_no_nan]
    OLDF_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in OLDF_lemma_cw_text_no_nan]
    PLD_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in PLD_lemma_cw_text_no_nan]
    PLDF_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in PLDF_lemma_cw_text_no_nan]
    subtlexus_log_freq_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in subtlexus_log_freq_lemma_cw_text_no_nan]
    subtlexus_log_cd_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in subtlexus_log_cd_lemma_cw_text_no_nan]
    coca_maga_cd_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in coca_maga_cd_lemma_cw_text_no_nan]
    coca_mag_log_freq_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in coca_mag_log_freq_lemma_cw_text_no_nan]
    num_rhymes_full_elp_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_rhymes_full_elp_lemma_cw_text_no_nan]
    num_rhymes_1000_coca_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_rhymes_1000_coca_lemma_cw_text_no_nan]
    num_rhymes_2500_coca_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_rhymes_2500_coca_lemma_cw_text_no_nan]
    num_rhymes_5000_coca_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_rhymes_5000_coca_lemma_cw_text_no_nan]
    num_rhymes_10000_coca_lemma_cw = [sum(sub_list) / len(sub_list) if sub_list else 0 for sub_list in num_rhymes_10000_coca_lemma_cw_text_no_nan]

    if "Basic counts" in cands:
        df2 = df2.assign(
            num_syl_lemma_cw = num_syl_lemma_cw,
            num_let_lemma_cw = num_let_lemma_cw,
            num_phone_lemma_cw = num_phone_lemma_cw,
            discrepancy_raw_lemma_cw = discrepancy_raw_lemma_cw,
            discrepancy_ratio_lemma_cw = discrepancy_ratio_lemma_cw,
            avg_syl_length_lemma_cw = avg_syl_length_lemma_cw,
            num_cons_char_lemma_cw = num_cons_char_lemma_cw,
            num_vowel_char_lemma_cw = num_vowel_char_lemma_cw,
            num_cons_phone_lemma_cw = num_cons_phone_lemma_cw,
            num_vowel_phone_lemma_cw = num_vowel_phone_lemma_cw,
            avg_phone_per_char_cons_lemma_cw = avg_phone_per_char_cons_lemma_cw,
            avg_phone_per_char_vowel_lemma_cw = avg_phone_per_char_vowel_lemma_cw,
            avg_phone_per_char_all_lemma_cw = avg_phone_per_char_all_lemma_cw,
            number_phone_vowel_lemma_cw = number_phone_vowel_lemma_cw,
            number_phone_all_lemma_cw = number_phone_all_lemma_cw,
            )

    if "Conditional probability" in cands:
        df2 = df2.assign(
            reverse_prior_prob_cons_lemma_cw = prior_prob_cons_lemma_cw,
            max_prob_cons_lemma_cw = max_prob_cons_lemma_cw,
            min_prob_cons_lemma_cw = min_prob_cons_lemma_cw,
            mid_prob_cons_lemma_cw = mid_prob_cons_lemma_cw,
            number_phone_cons_lemma_cw = number_phone_cons_lemma_cw,
            reverse_prior_prob_vowel_lemma_cw = prior_prob_vowel_lemma_cw,
            max_prob_vowel_lemma_cw = max_prob_vowel_lemma_cw,
            min_prob_vowel_lemma_cw = min_prob_vowel_lemma_cw,
            mid_prob_vowel_lemma_cw = mid_prob_vowel_lemma_cw,
            reverse_prior_prob_all_lemma_cw = prior_prob_all_lemma_cw,
            max_prob_all_lemma_cw = max_prob_all_lemma_cw,
            mid_prob_all_lemma_cw = mid_prob_all_lemma_cw,
            min_prob_all_lemma_cw = min_prob_all_lemma_cw,
            Conditional_Probability_Average_lemma_cw = Conditional_Probability_Average_lemma_cw,
            )

        df2['weight_max_prob_cons_lemma_cw'] = df2['max_prob_cons_lemma_cw'] / df2['number_phone_cons_lemma_cw']
        df2['weight_min_prob_cons_lemma_cw'] = df2['min_prob_cons_lemma_cw'] / df2['number_phone_cons_lemma_cw']
        df2['weight_mid_prob_cons_lemma_cw'] = df2['mid_prob_cons_lemma_cw'] / df2['number_phone_cons_lemma_cw']

        df2['weight_max_prob_vowel_lemma_cw'] = df2['max_prob_vowel_lemma_cw'] / df2['number_phone_vowel_lemma_cw']
        df2['weight_min_prob_vowel_lemma_cw'] = df2['min_prob_vowel_lemma_cw'] / df2['number_phone_vowel_lemma_cw']
        df2['weight_mid_prob_vowel_lemma_cw'] = df2['mid_prob_vowel_lemma_cw'] / df2['number_phone_vowel_lemma_cw']

        df2['weight_max_prob_all_lemma_cw'] = df2['max_prob_all_lemma_cw'] / df2['number_phone_all_lemma_cw']
        df2['weight_mid_prob_all_lemma_cw'] = df2['mid_prob_all_lemma_cw'] / df2['number_phone_all_lemma_cw']
        df2['weight_min_prob_all_lemma_cw'] = df2['min_prob_all_lemma_cw'] / df2['number_phone_all_lemma_cw']

    if "Neighborhood effects" in cands:
        df2 = df2.assign(
            Ortho_N_lemma_cw = Ortho_N_lemma_cw,
            Phono_N_lemma_cw = Phono_N_lemma_cw,
            Phono_N_H_lemma_cw = Phono_N_H_lemma_cw,
            OG_N_lemma_cw = OG_N_lemma_cw,
            OG_N_H_lemma_cw = OG_N_H_lemma_cw,
            Freq_N_lemma_cw = Freq_N_lemma_cw,
            Freq_N_P_lemma_cw = Freq_N_P_lemma_cw,
            Freq_N_PH_lemma_cw = Freq_N_PH_lemma_cw,
            Freq_N_OG_lemma_cw = Freq_N_OG_lemma_cw,
            Freq_N_OGH_lemma_cw = Freq_N_OGH_lemma_cw,
            OLD_lemma_cw = OLD_lemma_cw,
            OLDF_lemma_cw = OLDF_lemma_cw,
            PLD_lemma_cw = PLD_lemma_cw,
            PLDF_lemma_cw = PLDF_lemma_cw,
            )

    if "Word frequency" in cands:
        df2 = df2.assign(
            subtlexus_log_freq_lemma_cw = subtlexus_log_freq_lemma_cw,
            subtlexus_log_cd_lemma_cw = subtlexus_log_cd_lemma_cw,
            coca_maga_cd_lemma_cw = coca_maga_cd_lemma_cw,
            coca_mag_log_freq_lemma_cw = coca_mag_log_freq_lemma_cw,
            )

    if "Rhymes" in cands:
        df2 = df2.assign(
            num_rhymes_full_elp_lemma_cw = num_rhymes_full_elp_lemma_cw,
            num_rhymes_1000_coca_lemma_cw = num_rhymes_1000_coca_lemma_cw,
            num_rhymes_2500_coca_lemma_cw = num_rhymes_2500_coca_lemma_cw,
            num_rhymes_5000_coca_lemma_cw = num_rhymes_5000_coca_lemma_cw,
            num_rhymes_10000_coca_lemma_cw = num_rhymes_10000_coca_lemma_cw,
            )


    return df2

#====================================================================================


# For threading (so that the window doesn't freeze up when the code is running)
class Worker(QObject):

    def __init__(self, essays, file_name_v, op, testnames):
        super(Worker, self).__init__()
        self.procdocs = []
        self.opres = op
        self.essays = essays
        self.file_name_v = file_name_v
        self.testnames = testnames

    statusreport = pyqtSignal(str)
    finished = pyqtSignal()

    def run(self):
        #Append the output list with metrics for each text. Measurements are done in the minicheck function.
        self.statusreport.emit(f'Opening texts')
        for i in self.essays:
            try:
                # load text from a file as a string.
                with open(i, 'r', encoding='utf-8', errors='ignore') as f:
                    fl = f.read().lower()
                # process each word using spacy to prep
                self.procdocs.append(" ".join([''.join([y for y in list(x) if y.isalnum()]) for x in fl.split()]))
            except FileNotFoundError:
                self.procdocs.append("")

        self.statusreport.emit(f'Spacy is processing {len(self.essays)} texts... Please be patient...')
        self.procdocs = list(proc.pipe(self.procdocs))
        self.statusreport.emit(f'Spacy processing completed')

        df = pd.DataFrame({"target_files": self.essays})
        miniout = minicheck(df, self.procdocs, self.testnames)

        miniout.to_csv(os.path.join(os.getcwd(), f'{self.file_name_v.text()}.csv'), index=False)

        print(os.path.join(os.getcwd(), f'{self.file_name_v.text()}.csv'))
        self.statusreport.emit(f'Done! File was saved as {self.file_name_v.text()}.csv')

#====================================================================================

# Main Window: can hold status-bars and file menus if needed later
class MainWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('TAADA')
        self.mainwidge = InternalWidget()
        self.setCentralWidget(self.mainwidge)
        self.show()


class InternalWidget(QFrame):
    def __init__(self):
        super().__init__()

        # Check Boxes
        self.awl_box = QCheckBox("All words lemmatized")
        self.awul_box = QCheckBox("All words unlemmatized")
        self.cwl_box = QCheckBox("Content words lemmatized")
        self.cwul_box = QCheckBox("Content words unlemmatized")

        self.basic_counts_box = QCheckBox("Basic counts")
        self.cond_prob_box = QCheckBox("Conditional probability")
        self.neighborhood_box = QCheckBox("Neighborhood effects")
        self.word_freq_box = QCheckBox("Word frequency")
        self.rhyme_box = QCheckBox("Rhymes")

        self.initUI()

        # Files chosen will be appended here
        self.patients = []


    def main_execution(self, directory):
        try:
            self.thread.quit()
            self.thread.wait()
        except Exception as e:
            pass

        #Fetch texts from user's directory
        essays = directory
        #Initialize an output list with headers. The first entry in the list is the list of index headers for the csv.
        op = ['Filename']
        cands = []

        if self.awl_box.checkState():
            cands.append("All words lemmatized")
        if self.awul_box.checkState():
            cands.append("All words unlemmatized")
        if self.cwl_box.checkState():
            cands.append("Content words lemmatized")
        if self.cwul_box.checkState():
            cands.append("Content words unlemmatized")
        if self.basic_counts_box.checkState():
            cands.append("Basic counts")
        if self.cond_prob_box.checkState():
            cands.append("Conditional probability")
        if self.neighborhood_box.checkState():
            cands.append("Neighborhood effects")
        if self.word_freq_box.checkState():
            cands.append("Word frequency")
        if self.rhyme_box.checkState():
            cands.append("Rhymes")

        output = [op]

        # Step 2: Create a QThread object
        self.thread = QThread()
        # Step 3: Create a worker object
        self.worker = Worker(essays, self.file_name_v, output, cands)
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.statusreport.connect(self.process.setText)
        # Step 6: Start the thread
        self.thread.start()

    def initUI(self):
        # Initiate grid
        grid = QGridLayout()
        grid.setSpacing(15)
        self.setLayout(grid)

        # Create font types
        bold_and_whatnot = QFont()
        bold_and_whatnot.setBold(True)

        logo = QLabel()
        logos = ['Des1.png', "Des2.png", "Des3.png", "Des4.png"]
        qfm = QPixmap(resource_path(random.sample(logos,1)[0]))
        qfm = qfm.scaledToWidth(216)
        logo.setPixmap(qfm)
        logo.setAlignment(QtCore.Qt.AlignCenter)

        """CHOOSE FILE"""
        # Choose FIle - Instruction
        inst_file_choose = QLabel()
        inst_file_choose.setText('--Step 1 - Choose File(s)--')
        inst_file_choose.setFont(bold_and_whatnot)
        inst_file_choose.setAlignment(QtCore.Qt.AlignCenter)
        # Choose File - button
        file_choose = QPushButton('Choose File(s)')
        file_choose.clicked.connect(self.openFileNameDialog)
        # Show File Path ?

        # Choose Tests - Instructions
        inst_choose_lvl_a = QLabel()
        inst_choose_lvl_a.setText('--Step 2 - Choose Levels of Analysis--')
        inst_choose_lvl_a.setFont(bold_and_whatnot)
        inst_choose_lvl_a.setAlignment(QtCore.Qt.AlignCenter)

        # Choose Tests - Instructions
        inst_choose_test = QLabel()
        inst_choose_test.setText('--Step 3 - Choose Tests--')
        inst_choose_test.setFont(bold_and_whatnot)
        inst_choose_test.setAlignment(QtCore.Qt.AlignCenter)

        # Choose file name
        inst_choose_filename = QLabel()
        inst_choose_filename.setText('--Step 4 - Save Results as--')
        inst_choose_filename.setFont(bold_and_whatnot)
        inst_choose_filename.setAlignment(QtCore.Qt.AlignCenter)

        # Run Tests - Choose File Name
        self.file_name_v = QLineEdit()
        self.file_name_v.setText(f'Replace this with file name')

        # Check all checkboxes
        allcheck = QPushButton('Select all tests and levels')
        allcheck.clicked.connect(self.allchecker)

        # Run Tests - Instructions
        inst_run_test = QLabel()
        inst_run_test.setText('--Step 5 - Run Tests--')
        inst_run_test.setFont(bold_and_whatnot)
        inst_run_test.setAlignment(QtCore.Qt.AlignCenter)

        # Run Tests
        run_test = QPushButton('Run Tests')
        run_test.clicked.connect(lambda: self.main_execution(self.patients))

        # Progress bar
        self.process = QLineEdit()
        self.process.setReadOnly(True)
        self.process.setText('Waiting...')

        # Open file
        open_result_csv = QPushButton("Open Result File")
        open_result_csv.clicked.connect(self.open_result_csv_func)
        
        # Add Widgets (row, col, #rowspan, #colspan)
        grid.addWidget(logo,0,0,1,2)
        grid.addWidget(inst_file_choose, 20, 0, 1, 2)
        grid.addWidget(file_choose, 21, 0, 1, 2)

        grid.addWidget(inst_choose_lvl_a, 22, 0, 1, 2)
        grid.addWidget(self.awl_box, 23, 0)
        grid.addWidget(self.awul_box, 23, 1)
        grid.addWidget(self.cwl_box, 24, 0)
        grid.addWidget(self.cwul_box, 24, 1)

        grid.addWidget(inst_choose_test, 27, 0, 1, 2)
        grid.addWidget(self.basic_counts_box, 28, 0)
        grid.addWidget(self.cond_prob_box, 28, 1)
        grid.addWidget(self.neighborhood_box, 29, 0)
        grid.addWidget(self.word_freq_box, 29, 1)
        grid.addWidget(self.rhyme_box, 30, 0)

        grid.addWidget(allcheck, 32, 0, 1, 2)
        grid.addWidget(inst_choose_filename, 33, 0, 1, 2)
        grid.addWidget(self.file_name_v, 34, 0, 1, 2)
        grid.addWidget(inst_run_test, 35, 0, 1, 2)
        grid.addWidget(run_test, 36, 0, 1, 2)
        grid.addWidget(self.process, 37, 0, 1, 2)

        grid.addWidget(open_result_csv, 38, 0, 1, 2)

    def open_result_csv_func(self):
        try:
            subprocess.Popen(['start', '', os.path.join(os.getcwd(), f'{self.file_name_v.text()}.csv')], shell=True)
            self.process.setText('Opened csv file')
        except Exception as e:
            self.process.setText('Unable to open csv file')

    def allchecker(self):
        self.awl_box.setChecked(True)
        self.awul_box.setChecked(True)
        self.cwl_box.setChecked(True)
        self.cwul_box.setChecked(True)

        self.basic_counts_box.setChecked(True)
        self.cond_prob_box.setChecked(True)
        self.neighborhood_box.setChecked(True)
        self.word_freq_box.setChecked(True)
        self.rhyme_box.setChecked(True)

    # Open file window (:connected to file_choose button)
    def openFileNameDialog(self):
        fileName, _ = QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "","Text Files (*.txt)")
        if fileName:
            self.patients = []
            for item in fileName:
                self.patients.append(item)
                self.process.setText(f'You have chosen {len(fileName)} files')

# QT style sheet to make things look better
style = """
        QFrame {
        background-color: #FFFFFF;
        }

        QPushButton {
            background-color: #e8f4ea;
            border: 1px solid #dbdbdb;
            border-radius: .175em;
            color: #363636;
        }
        QPushButton:pressed {
            background-color: #b8d8be;
        }
        QCheckBox {
            color: #352424;
        }
        QLineEdit {
            border-width: 1px;
            border-color: #e8f4ea;
            border-style: outset;
            padding: 1.5px;
            background-color: #e0f0e3;
        }
"""

if __name__ == '__main__':
    app = QApplication(sys.argv)
    try:
        app.setWindowIcon(QIcon(resource_path('taada_logo.ico')))
    except:
        pass
    app.setStyleSheet(style)
    ex = MainWidget()
    sys.exit(app.exec_())
