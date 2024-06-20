import os

DATASETS = {
    "it" : "Paul/hatecheck-italian",
    "en" : "Paul/hatecheck",
    "fr" : "Paul/hatecheck-french",
    "es" : "Paul/hatecheck-spanish",
    "de" : "Paul/hatecheck-german",
}

MODELS = {
    "it" : "Geotrend/distilbert-base-it-cased",
    "en" : "Geotrend/distilbert-base-en-cased",
    "fr" : "Geotrend/distilbert-base-fr-cased",
    "es" : "Geotrend/distilbert-base-es-cased",
    "de" : "Geotrend/distilbert-base-de-cased",
    "multi"  : "distilbert/distilbert-base-multilingual-cased"
}

TOKENIZER_CONFIG = os.path.join("configs", "tokenizer_config.json")


MODEL_CONFIGS = {
    "Geotrend/distilbert-base-it-cased" : os.path.join("configs", "it_model_config.json"), 
    "Geotrend/distilbert-base-en-cased" : os.path.join("configs", "en_model_config.json"),
    "Geotrend/distilbert-base-fr-cased" : os.path.join("configs", "fr_model_config.json"),
    "Geotrend/distilbert-base-es-cased" : os.path.join("configs", "es_model_config.json"),
    "Geotrend/distilbert-base-de-cased" : os.path.join("configs", "de_model_config.json"),
    "distilbert/distilbert-base-multilingual-cased" : os.path.join("configs", "multi_model_config.json"),
}

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

HATE_TYPES = {
    "LGBT" : {"trans people", "gay people"},
    "GENERIC" : {},
    "RELIGIOUS" :{"Muslims", "jews"},
    "RACIAL" : {"black people"},
    "MISOGYNY" : {"women"},
    "DISABILITY" : {"disabled people"},
    "XENOPHOBIC" : {"immigrants", "refugees", "indigenous people"},
}

ID2LABEL_B = { 
    0 : "NEGATIVE", 
    1 : "NON-HATEFUL"
}

LABEL2ID_B = {
    "NEGATIVE": 0, 
    "NON-HATEFUL" : 1
}

ID2LABEL_M = { 
    0 : "LGBT", 
    1 : "GENERIC",
    2 : "RELIGIOUS",
    3 : "RACIAL",
    4 : "MISOGENIC",
    5 : "DISABILITY",
    6 : "XENOPHOBIC",
}

LABEL2ID_M = {
    "LGBT" : 0, 
    "GENERIC" : 1,
    "RELIGIOUS" : 2,
    "RACIAL" : 3,
    "MISOGENIC" : 4,
    "DISABILITY" : 5,
    "XENOPHOBIC" : 6,
}