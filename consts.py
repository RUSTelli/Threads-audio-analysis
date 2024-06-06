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

NEUTRAL = {
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
CUSTOM_FUNCTIONALITIES = {
    "MENACE" : MENACE,
    "GENERIC_HATE" : GENERIC_HATE,
    "CATEGORY_HATE" : CATEGORY_HATE,
    "VOLGARITY" : VOLGARITY,
    "NEUTRAL" : NEUTRAL,
    "ANTI_HATE" : ANTI_HATE,
    "SUPPORTING" : SUPPORTING,
    "POSITIVE" : POSITIVE,
}

# atm this is not programmed, but let's keep it here for a 2 class classification.
'''
SENTIMENT_CLUSTER = {
    "HATE" : [MENACE, GENERIC_HATE, CATEGORY_HATE,],
    "POSITIVE" : [ANTI_HATE, SUPPORTING, POSITIVE,],
}
'''