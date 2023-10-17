import copy
import re

import ftlangdetect.detect as detect_ft
from cleantext import clean
from langdetect import detect_langs
from tqdm import tqdm


def cleanbasic(tweets):
    '''
    Cleaning using the cleantext library and re library
    :return: list of cleaned tweets
    '''

    print("Cleaning using the cleantext library...")
    num_before = len(tweets)
    clean_tweets = []

    # for tweet_id in (pbar := tqdm(tweets)):
    #     pbar.set_description(f"clean-text")
    for tweet_id in tqdm(tweets):

        clean_tweet = re.sub("@[A-Za-z0-9_]+", "", tweets[tweet_id]["text"])
        clean_tweet = re.sub("#[A-Za-z0-9_]+", "", clean_tweet)

        clean_tweet = clean(clean_tweet,
                            lower=False,
                            no_urls=True,
                            no_emoji=True,
                            no_emails=True,
                            no_phone_numbers=True,
                            # no_numbers=True,
                            lang='de'
                            )
        # Remove mentions and hashtags too
        tweets[tweet_id]["text"] = clean_tweet

    num_after = len(tweets)
    # print(f"Removed {100*(num_before-num_after)/num_before}% of tweets")

    return tweets

def cleanrem(tweets):
    '''
    List of tweets with words with special characters removed
    :return: list of tweets
    '''

    print("Removing words with special characters...")
    num_before = len(tweets)

    remove = ['@', '*']
    clean_tweets = []

    # for tweet_id in (pbar := tqdm(tweets)):
    #     pbar.set_description(f"clean-spec")
    for tweet_id in tweets:
        tweet = str(tweets[tweet_id]["text"])
        words = tweet.split()
        words_clean = []

        for w in words:
            if not any((r in w) for r in remove):
                words_clean.append(w)
        clean_tweet = ' '.join(words_clean)

        tweets[tweet_id]['text'] = clean_tweet

    num_after = len(tweets)
    # print(f"Removed {100*(num_before-num_after)/num_before}% of tweets")
    
    return tweets

def cleanlan(tweets):
    '''
    Method to remove tweets that do not identify as German after cleaning
    :return: list of tweets
    '''
    print("Removing non-german tweets...")

    clean_tweets = []

    num_before = len(tweets)
    
    # for tweet_id in (pbar := tqdm(list(tweets))):
    #     pbar.set_description(f"clean-lang")
    for tweet_id in copy.deepcopy(tweets):
        tweet_prim = str(tweets[tweet_id]["text"])
        tweet_prim = re.sub('".*?"', '', tweet_prim)
        tweet_prim = re.sub("[\(\<].*?[\)\>]", "", tweet_prim).strip()

        # Remove short tweets due to poor language detection
        if len(tweet_prim.split()) > 7:
            try:
                detection = detect_langs(tweet_prim)

                detected_langs = [i.lang for i in detection]
                both_langs_there = 'en' in detected_langs and 'de' in detected_langs
                neither_main_lang = detection[0].lang != 'de' and detection[0].lang != 'en'
                other_strong_second = len(detection) > 1 and (detection[1].lang != 'de' and detection[1].lang != 'en') and (detection[1].prob > 0.3)
                
                if both_langs_there:
                    tweets[tweet_id]["text"] = tweet_prim
                elif neither_main_lang or other_strong_second:
                    tweets.pop(tweet_id)
                else:
                    tweets[tweet_id]["text"] = tweet_prim
                # else:
                #     tweets.pop(tweet_id)
            except:
                tweets.pop(tweet_id)
                continue
        else:
            tweets.pop(tweet_id)

    num_after = len(tweets)
    print(f"Removed {100*(num_before-num_after)/num_before}% of tweets")
    return tweets

def clean_lingua(tweets):
    languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH, Language.DANISH,
                 Language.FINNISH, Language.BOKMAL, Language.NYNORSK, Language.SWEDISH, Language.AFRIKAANS, Language.ITALIAN]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()

    for tweet_id in tqdm(tweets.copy()):

        tweet = re.sub("[\(\<].*?[\)\>]", "", tweets[tweet_id]['text']).strip()

        main_lan = detector.detect_language_of(tweet)
        all_lans = detector.detect_multiple_languages_of(tweet)
        confidence_values = detector.compute_language_confidence_values(tweet)

        found1 = False
        found2 = False

        if main_lan.name != 'GERMAN':
            print(f"Exiting, as main language is {main_lan.name}")
            tweets.pop(tweet_id)
            continue

        for language, value in confidence_values:
            if not (language.name == 'GERMAN') or (language.name == 'ENGLISH'):
                if value > 0.05:
                    tweets.pop(tweet_id)
                    print(f"Exiting as {language.name} has probability {value}")
                    found1 = True
                    break

        if found1: continue

        for result in all_lans:
            if not (result.language.name == 'GERMAN' or result.language.name == 'ENGLISH'):
                print(f"{result.language.name}: '{tweet[result.start_index:result.end_index]}'")
                print(f"Exiting as there is some {result.language.name} in there...")
                tweets.pop(tweet_id)
                found2 = True
                break
    
        if found2: continue
    
    return tweets

def cleanfastt(tweets):
    print("Cleaning with FastText")
    num_before = len(tweets)
    for tweet_id in copy.deepcopy(tweets):
        tweet = re.sub("[\(\<].*?[\)\>]", "", tweets[tweet_id]['text']).strip()
        tweet = tweet.replace("  ", " ")
        result = detect_ft(text=tweet, low_memory=False)
        if not (result['lang'] == 'de' or result['lang'] == 'en'):
            tweets.pop(tweet_id)
    num_after = len(tweets)
    print(f"Removed {100*(num_before-num_after)/num_before}% of tweets")
    return tweets

def cleanempty(tweets):
    print("Removing empty tweets")
    for tweet_id in copy.deepcopy(tweets):
        if len(tweets[tweet_id]['text'].split()) < 4:
            tweets.pop(tweet_id)

    return tweets

# def stats(tweets):
#     '''
#     Method to generate and print some basic stats about how many tweets kept
#     '''
#     numcleantweets = len(tweets)
#     perc = (numcleantweets / numtweets) * 100
#     # print("STATS REPORT:")
#     print(f"\n {perc}% of tweets were kept from {self.tweets_file} (that is {self.numcleantweets} out of {self.numtweets})")
