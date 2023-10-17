
tot_b = 0
tot_a = 0


class Preprocessing:
    ''' Preprocessing of each tweet in a month and save cleaned tweets in text file
    Removes words containing particular characters (e.g. @ for tweets)
    Performs a standard sentence cleaning protocol from cleantext
    Ensures saved tweets are identified as German after cleaning
    '''

    def __init__(self, config, tweets):
        self.config = config
        self.tweets = tweets['zenodo_tweets']

    def media_cleaning(self):
        '''
        Cleaning pipline: cleaning, language detection and removal
        '''
        for tool in self.config.cleaning:
            getattr(self, tool)()

        # self.stats()

