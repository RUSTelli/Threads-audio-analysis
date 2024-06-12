BINARY_CLUSTER = {
    "HATEFUL" : {
        "threat_dir_h",
        "threat_norm_h",
        "target_obj_nh",
        "target_indiv_nh",
        "target_group_nh",
        "derog_neg_emote_h",
        "derog_neg_attrib_h",
        "derog_dehum_h",
        "derog_impl_h",
        "slur_h",
        "ref_subs_clause_h",
        "ref_subs_sent_h",
        "negate_pos_h",
        "phrase_question_h",
        "phrase_opinion_h",
        "profanity_h",
    },

    "NON-HATEFUL" : {
        "profanity_nh",
        "ident_neutral_nh",
        "counter_quote_nh",
        "counter_ref_nh",
        "negate_neg_nh",
        "ident_pos_nh",
    },
}

# SENTIMENT_CLUSTER = {
#     "HATEFUL" : { 
#         "threat_dir_h",
#         "threat_norm_h",
#         "target_obj_nh",
#         "target_indiv_nh",
#         "target_group_nh",
#         "derog_neg_emote_h",
#         "derog_neg_attrib_h",
#         "derog_dehum_h",
#         "derog_impl_h",
#         "slur_h",
#         "ref_subs_clause_h",
#         "ref_subs_sent_h",
#         "negate_pos_h",
#         "phrase_question_h",
#         "phrase_opinion_h",
#         "profanity_h",
#     },
#     "NEUTRAL" : {
#         "profanity_nh",
#         "ident_neutral_nh",
#     },
#     "POSITIVE" : {
#         "counter_quote_nh",
#         "counter_ref_nh",
#         "negate_neg_nh",
#         "ident_pos_nh",
#     },
# }



HATE_TYPES = {
    "LGBT" : {"trans people", "gay people"},
    "GENERIC" : {},
    "RELIGIOUS" :{"Muslims"},
    "RACIAL" : {"black people"},
    "SEXUAL" : {"women"},
    "DISABILITY" : {"disabled people"},
    "XENOPHOBIC" : {"immigrants"},
}
# TODO restore origianl terniary labels

# ID2LABEL = {0: "NEGATIVE", 1: "NEUTRAL", 2 : "POSITIVE"}

# LABEL2ID = {"NEGATIVE": 0, "NEUTRAL" : 1,"POSITIVE": 2}

ID2LABEL = { 
    0 : "NEGATIVE", 
    1 : "NON-HATEFUL"
}

LABEL2ID = {
    "NEGATIVE": 0, 
    "NON-HATEFUL" : 1
}


'''
MENACE = {
    "threat_dir_h",
    "threat_norm_h",
}

GENERIC_HATE = {
    "target_obj_nh",
    "target_indiv_nh",
    "target_group_nh",
}

CATEGORY_HATE = {
    "derog_neg_emote_h", 
    "derog_neg_attrib_h", 
    "derog_dehum_h", 
    "derog_impl_h",
    "slur_h",     
    "ref_subs_clause_h",
    "ref_subs_sent_h",
    "negate_pos_h",
    "phrase_question_h",
    "phrase_opinion_h",
    "profanity_h",
}

VOLGARITY = {
    "profanity_nh",
}

NEUTRALITY = {
    "ident_neutral_nh",
}

ANTI_HATE = {
    "counter_quote_nh",
    "counter_ref_nh"
        
}

SUPPORTING = {
    "negate_neg_nh",
}

POSITIVE = {
    "ident_pos_nh",
}

SENTIMENT_CLUSTER = {
    "HATEFUL" : [MENACE, GENERIC_HATE, CATEGORY_HATE,],
    "NEUTRAL" : [VOLGARITY, NEUTRALITY,],
    "POSITIVE" : [ANTI_HATE, SUPPORTING, POSITIVE,],
}
'''


