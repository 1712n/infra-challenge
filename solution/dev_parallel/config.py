config = { 

    
    "workers": [

        {
            "queue": 1,
            "worker_name": "worker1",
            "model_name": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
            "model_labels": ["NEGATIVE", "NEUTRAL", "POSITIVE"] # Labels for sentiment model
        },
        {
            "queue": 2,
            "worker_name": "worker2",
            "model_name": "ivanlau/language-detection-fine-tuned-on-xlm-roberta-base",
            "model_labels": ["Arabic", "Basque", "Breton", "Catalan", "Chinese_China", "Chinese_Hongkong", "Chinese_Taiwan", "Chuvash", "Czech", "Dhivehi", "Dutch", "English", "Esperanto", "Estonian", "French", "Frisian", "Georgian", "German", "Greek", "Hakha_Chin", "Indonesian", "Interlingua", "Italian", "Japanese", "Kabyle", "Kinyarwanda", "Kyrgyz", "Latvian", "Maltese", "Mongolian", "Persian", "Polish", "Portuguese", "Romanian", "Romansh_Sursilvan", "Russian", "Sakha", "Slovenian", "Spanish", "Swedish", "Tamil", "Tatar", "Turkish", "Ukrainian", "Welsh"] # Labels for language detection model
        },
        {
            "queue": 3,
            "worker_name": "worker3",
            "model_name": "svalabs/twitter-xlm-roberta-crypto-spam",
            "model_labels": ["HAM", "SPAM"] # Labels for crypto spam model
        },
        {
            "queue": 4,
            "worker_name": "worker4",
            "model_name": "EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus",
            "model_labels": ["LABEL_0", "LABEL_1"] # Label_1 means TOXIC, Label_0 means NOT TOXIC.
        },
        {
            "queue": 5,
            "worker_name": "worker5",
            "model_name": "jy46604790/Fake-News-Bert-Detect",
            "model_labels": ["LABEL_0", "LABEL_1"] # LABEL_0: Fake news, LABEL_1: Real news
        }
    ]
}
