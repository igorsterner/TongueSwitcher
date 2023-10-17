#44 most frquent English ngrams (bi/tri)
#https://www3.nd.edu/~busiforc/handouts/cryptography/Letter%20Frequencies.html
MOST_COMMON_NGRAMS_EN = ["th","he","in","en","nt","re","er","an","ti","es","on","at","se","nd","or","ar","al","te","co","de","to","ra","et","ed","it","sa","em","ro"]+["the","and","tha","ent","ing","ion","tio","for","nde","has","nce","edt","tis","oft","sth","men"]
#60 most frequent German ngrams (bi/tri)
#http://practicalcryptography.com/cryptanalysis/letter-frequencies-various-languages/german-letter-frequencies/
MOST_COMMON_NGRAMS_DE = ["er","en","ch","de","ei","te","in","nd","ie","ge","st","ne","be","es","un","re","an","he","au","ng","se","it","di","ic","sc","le","da","ns","is","ra"]+["der","ein","sch","ich","nde","die","che","den","ten","und","ine","ter","gen","end","ers","ste","cht","ung","das","ere","ber","ens","nge","rde","ver","eit","hen","erd","rei","ind"]


class FunctionWords:
    """contains lists of function words"""

    _D_article = ["der","die","das","dem","den","des"]+["ein","eine","einem","einer","einen","eines"]
    _D_pronouns = ["ich","du","er","sie","es","wir","ihr","man"
                   "mich","dich","sich","ihn","uns","euch",
                   "mir","dir","ihm","ihr"] \
                  + ["mein","meine","meiner","meins",
                     "dein","deine","deiner","deins",
                     "sein", "seine", "seiner", "seins",
                     "ihr", "ihre", "ihrer", "ihrs",
                     "unser","unsere","unseres","unserer","unsers", "unsre", "unsres", "unsrer",
                     "euer","euere","euerer","eueres","euers","eure""eurer","eures"] \
                  + ["diese","dieser","dieses","diesen","diesem"
                      ,"jene","jener","jenes","jenem","jenen"]\
                  + ["irgendein","irgendeine","irgendeiner","irgendeins","irgendeinem","irgendeinen",
                   "irgendjemand","irgendjemanden","irgendjemandes","irgendjemandem",
                     "irgendwelche","irgendwelcher","irgendwelchem","irgendwelchen","irgendwelches",
                    "welche","welcher","welchem","welchen","welches",
                   "jemand","jemanden","jemandes","jemandem",
                     "etwas","irgendetwas","irgendwas"
                   "irgendwas","irgendetwas",
                     "niemand","nichts"]+\
                  ["irgendwie","irgendwo","irgendwann",
                   "nirgendwo","nirgends","nie"]
    _D_prepositions = ["vor","vors","vorm","hinter","hinters","hinterm",
                       "über","übers","überm","unter","unterm","unters",
                       "neben","zwischen",
                       "außerhalb","innerhalb",
                       "aus","herein","hinein","heraus","hinaus","von","vom",
                       "zu","zur","zum","nach","in","an","ans","am","durch","auf","aufs"]+\
                      ["vor","nach","während","bis","ab"] +\
                      ["weil","da","daher","wegen","um"]+\
                      ["ums","außer","bei","beim","mit","ohne","gegen","für","fürs", "statt","trotz"]
    _D_conjunction = ["und", "oder", "aber", "weil",
                       "da","zumal", "denn","obwohl",
                       "dennoch","hingegen","sonst","außerdem",
                       "damit","ob","wenn","dass"]
    _D_questionwords = ["wer", "wie", "wo", "was", "wann","warum","weshalb","wieso","weswegen","wofür","woher","wohin", "wozu"]
    _D_particles = ["ja", "nein", "nicht", "kein", "sehr", "gern", "viel", "vieles", "viele", "vielen", "vieler", "vielem"]
    _D_timeplace = ["jetzt", "heute", "gestern", "morgen", "später", "früher", "spät", "früh",
                  "dann","danach","davor","vor","nach","während",
                  "oft","häufig","selten","nie","immer","regelmäßig","täglich",
                  "sofort","bald","wieder","irgendwann","nie"] + \
                   ["hier","dort","da","draußen","fort","drinnen","überall","nirgendwo","irgendwo","nirgends"] + \
                   ["also","darum","demnach","deshalb","folglich","somit","trotzdem"]
    _D_auxiliaries = ["haben", "habe", "hab", "hast", "hat", "habt", "hatte", "hatten", "hattest", "hattet", "gehabt"] \
                     + ["sein","bin","bist","ist","sind","seid","war","warst","war","waren","wart","waren","gewesen"] + \
                     ["werden","werde","wirst","wird","werden","werdet","wurden","wurde","wurdest","wurdet"]
    deu_function_words = _D_auxiliaries + _D_article + _D_pronouns + _D_prepositions + _D_conjunction + _D_questionwords + _D_particles + _D_timeplace #todo



    _E_article = ["a","an","the"]
    _E_pronouns = ["I","you","he","she","it","we","they",
                   "me","my","him","her","his","their","them",
                   "mine","yours","hers","theirs","ours",
                   "myself","yourself","herself","himself","ourselves"] +\
                  ["this","that","these","those"] + \
                  ["something","someone","somebody",
                   "nothing","noone","no-one","nobody",
                   "anything","anyone","anybody",
                   "everything","everyone","everybody",
                   "every","some","any","no",
                   "everywhere","somewhere","anywhere","nowhere",
                   "somehow","anyhow"]
    _E_prepositions = ["to","of","for","before","after","while",
                       "during","untill","until","from","because",
                       "into","onto","in","on",
                       "behind","between","above","over","underneath",
                       "under","next","inside","outside","from","at","with","without",
                       "against","instead"]
    _E_conjunction = ["and", "or", "but", "because", "for", "since",
                       "though","although","even",
                       "else","despite","if","whether","that"]
    _E_questionwords = ["who", "how", "where", "what", "when", "why"]
    _E_particles = ["yes", "no", "not", "none", "non", "very", "much", "many"]
    _E_adverbs = ["now", "later", "earlier", "late",
                  "early", "tomorrow", "yesterday",
                  "then",
                  "today", "before", "after", "ago", "while", "during",
                "always", "never", "sometimes", "often",
                  "rarely", "sometime", "just", "immediately"] \
                 + ["here", "there", "outside", "inside"] \
                 + ["therefore","therefor","however","hence","thus"]
    _E_auxiliaries = ["have", "has", "had",
                      "be","is","am","are","was","were","been"
                      "will","would"]
    eng_function_words = _E_auxiliaries + _E_article + _E_pronouns + _E_prepositions + _E_conjunction + _E_questionwords + _E_particles + _E_adverbs #todo



class FlexDeri:
    """contains derivation and flexion affixes"""

    D_DER_A_suf_dict = {"ig": ["ig", "ige", "iger", "iges", "igen", "igem"],
                        "lich": ["lich", "iche", "icher", "iches", "ichen", "ichem"],
                        "sam": ["sam", "same", "samer", "sames", "samen", "samem"],
                        "haft": ["haft", "hafte", "hafter", "haftes", "haften", "haftem"],
                        "bar": ["bar", "bare", "barer", "bares", "baren", "barem"],
                        "reich": ["reich", "reiche", "reicher", "reiches", "reichen", "reichem"],
                        "arm": ["arm", "arme", "armer", "armes", "armen", "armem"],
                        "voll": ["voll", "volle", "voller", "volles", "vollen", "vollem"],
                        "los": ["los", "lose", "loser", "loses", "losen", "losem"],
                        "isch": ["isch", "ische", "ischer", "isches", "ischen", "ischem"],
                        "frei": ["frei", "freie", "freier", "freies", "freien", "freiem"]
                        }

    D_DER_N_suf_dict = {"heit": ["heit", "heiten"], "keit": ["keit", "keiten"],
                        "schaft": ["schaft", "schaften"], "ung": ["ung", "ungen"],
                        "nis": ["nis", "nisse", "nissen"], "tum": ["tum", "tümer", "tümern"],
                        "tät": ["tät", "täten"]}

    D_DER_V_pref_list = ["ab", "an", "auf", "aus", "be", "bei", "da", "dar", "durch", "ein", "ent", "er",
                         "fort", "ge", "her", "hin", "hinter", "los", "mit", "nach", "nieder", "über",
                         "um", "un", "unter", "ur", "ver", "vor", "weg", "wieder", "zer", "zu", "zurück",
                         "zusammen", "zwischen"]

    E_DER_A_suf_list = ["ly", "ful", "able", "ible", "less", "al", "ish", "like", "ic", "ical", "ically", "ious", "ous"]

    E_DER_N_suf_dict = {"ity": ["ity", "ities"], "tion": ["tion", "tions", "ion", "ions", "ation", "ations"],
                        "logy": ["logy", "logies"], "ant": ["ant", "ants"], "hood": ["hood", "hoods"],
                        "ess": ["ess", "esses"], "ness": ["ness", "nesses"], "ism": ["ism", "isms"],
                        "ment": ["ment", "ments"],
                        "ist": ["ist", "ists"], "acy": ["acy", "acies"], "a/ence": ["ance", "ence", "ances", "ences"],
                        "dom": ["dom"]}

    E_DER_V_pref_list = ["a", "after", "back", "be", "by", "down", "en",
                         "em", "fore", "hind", "mid", "midi", "mini", "mis", "off",
                         "on", "out", "over", "self", "step", "twi", "un", "under",
                         "up", "with"]


    D_FLEX_V_suf_list = ["en", "et", "t", "e", "st", "est", "te"]
    D_DER_V_suf_dict = {"ieren": ["ieren", "iere", "ierst", "iert"]}
    D_FLEX_A_suf_dict = {"er": ["er", "ere", "erer", "eres", "eren", "erem"],
                         "ste": ["ste", "ster", "stes", "sten", "stem"],
                         "end": ["end", "ende", "ender", "endem", "enden", "endes"]}
    D_FLEX_N_suf_list = ["er", "ern", "en", "e"]

    E_FLEX_V_suf_list = ["ing", "ed", "s", "es"]
    E_DER_V_suf_dict = {"ize": ["ize", "ise", "izes", "ises", "ized", "ised"], "fy": ["fy", "fies", "fied"]}
    E_FLEX_A_suf_list = ["er", "est", "st", "nd", "rd", "th"]
    E_FLEX_N_suf_list = ["s"]



class NELexMorph:
    """contains morphologcal and lexical components of Named Entities"""

    D_NE_Demo_suff = ["isch", "ische", "ischen", "ischem", "ischer", "er", "ern", "ese", "esen"]
    D_NE_Morph_suff = ["ien"]
    E_NE_Demo_suff = ["\'s", "", "ic", "ish", "ian", "ians", "ean", "eans", "ese", "eses"]
    E_NE_Morph_suff = ["y", "ey"]


    O_NE_Morph_suff = ["ia", "a"]

    D_NE_parts = ["reich", "berg", "weiler", "stade", "stadt", "stein", "tal", "see", "haven", "hafen", "graben",
                  "land", "furt", "burg", "fels", "feld", "gmund", "sankt", "gmünd", "berg", "bach", "wald", "born",
                  "brück", "bühl", "dorf", "bad", "kirch", "hausen", "land", "heim",
                  "neu", "groß", "klein", "mittel", "nord", "süd", "ost", "west", "hoch", "nieder", "mann", "platz","straße"]

    E_NE_parts = ["spring", "ford", "forth", "dale", "chester", "cester", "caster", "minster", "ham", "ing", "shep",
                  "ship", "thorp", "thorpe", "mouth", "tun", "ton", "land", "isle", "island", "king", "queen", "borough", "brough",
                  "burgh", "pool", "port", "porth", "gate", "ley", "leigh", "wick", "wich", "wych", "wyke", "worth", "worthy",
                  "new", "north", "south", "east", "west", "nor", "ast", "sud", "sut", "wes",
                  "boulevard", "street", "avenue", "square"]

    O_NE_suff = ["stan", "bad", "polis"]


class NELists:
    """contains lists of Named Entities"""
    # GERMAN Regions
    _deu_countries = ["Deutschland", "Österreich", "Schweiz", "Liechtenstein"]

    _D_cities = ["Berlin", "Hamburg", "München", "Köln", "Frankfurt", "Stuttgart", "Düsseldorf", "Leipzig", "Dortmund",
                 "Essen", "Bremen", "Dresden", "Hannover", "Nürnberg", "Duisburg"]\
                +["Straßburg", "Danzig", "Königsberg", "Lemberg", "Breslau", "Stettin"]
    _Ö_cities = ["Wien", "Graz", "Linz", "Salzburg", "Innsbruck"]
    _CH_cities = ["Zürich", "Genf", "Basel", "Bern", "Luzern"]

    _deu_städte = _D_cities + _Ö_cities + _CH_cities

    _D_parts_berlin = ["Spandau", "Charlottenburg", "Willmersdorf", "Steglitz", "Zellendorf",
                            "Reinickendorf", "Mitte", "Pankow", "Tempelhof", "Schöneberg",
                            "Neukölln", "Treptow", "Köpenick", "Marzahn", "Hellersdorf",
                            "Lichtenberg", "Kreuzberg", "Friedrichshain"]
    _D_regions = ["Rheinland", "Pfalz", "Saarland", "Hessen", "Baden", "Württemberg",
                   "Bayern", "Thüringen", "Nordrhein-Westfalen", "Westfalen", "Niedersachsen",
                   "Sachsen-Anhalt", "Sachsen", "Brandenburg", "Mecklenburg", "Vorpommern",
                   "Schleswig", "Holstein"] \
                 + ["Rhein-Main", "Rhein-Neckar", "Ruhrgebiet", "Ruhrpott", "Ruhr", "Rhein",
                     "Main", "Neckar", "Franken", "Schwaben"]+\
                 ["Pommern", "Schlesien", "Elsass", "Elsaß", "Lothringen", "Preußen", "Posen"]
    _Ö_regions = ["Tirol", "Arlberg", "Kärnten", "Burgenland", "Steiermark"]
    _deu_regionen = _D_parts_berlin + _D_regions + _Ö_regions


    _D_regionen_abbr = ["NRW", "RLP", "SL", "BW", "BaWü", "BY", "BE", "BER", "BB", "HB", "HH", "MV", "RP", "MeckPom"]
    deu_reg_abkürzungen = _D_regionen_abbr

    _deu_geographie = ["Rhein", "Weser", "Elbe", "Donau", "Main", "Saale", "Spree", "Ems", "Neckar", "Havel", "Leine", "Isar",
                 "Aller", "Elster", "Lahn"]\
                      +["Zugspitze"]\
                      +["Sylt", "Rügen", "Usedom"]\
                      +["Bodensee"]


    D_NE_REGs = [word.lower() for word in _deu_städte + _deu_regionen + _deu_geographie]
    D_NE_REGs_abbr = [word.lower() for word in deu_reg_abkürzungen]



    # ENGLISH Regions
    _us_cities = ["New York", "NYC", "Los Angeles", "Chicago", "Houston", "Phoenix", "Dallas", "Houston", "Washington",
                  "Philadelphia",
                  "San Antonio", "San Diego", "San José", "Austin", "Jacksonville",
                  "Columbus", "Indianapolis", "Miami", "Atlanta", "Boston", "Phoenix",
                  "San Francisco", "Charlotte", "Seattle", "Denver", "Nashville",
                  "Boston", "Portland", "Las Vegas", "Detroit", "Memphis"]
    _uk_cities = ["London", "Birmingham", "Manchester", "Leeds, Bradford", "Glasgow", "Liverpool", "Sheffield",
                  "Nottingham", "Bristol", "Edinburg", "Leicester", "Coventry", "Cardiff", "Belfast"]
    _can_cities = ["Toronto", "Montreal", "Vancouver", "Calgary", "Edmonton", "Ottawa"]
    _aus_cities = ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide", "Canberra"]
    _ire_cities = ["Dublin"]

    _us_states = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
                  "Florida",
                  "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
                  "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
                  "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
                  "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma",
                  "Oregon",
                  "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah",
                  "Vermont",
                  "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming", "District of Columbia", "Columbia"]

    _uk_regions = ["Yorkshire", "Midlands", "Humber", "Wales", "Scotland", "Ireland"]

    _aus_regions = ["Queensland", "Victoria", "Tasmania"]
    _eng_cities = _us_cities + _can_cities + _aus_cities + _uk_cities + _ire_cities

    _eng_regions = _us_states + _uk_regions + _aus_regions

    E_NE_REGs = [word.lower() for word in _eng_cities + _eng_regions]

    _eng_regions_abbr = ["DC","NYC"]
    E_NE_REGs_abbr = [word.lower() for word in _eng_regions_abbr]

    # OTHER Regions
    _eu_cities = ["Paris", "Madrid, Rom", "Rome", "Roma", "Warschau", "Warsaw", "Amsterdam", "den Haag", "the Hague",
                  "Athen", "Athens", "Bratislava", "Bratislawa", "Brüssel", "Brussels", "Budapest", "Bukrarest",
                  "Bucharest" "Helsinki", "Kopenhagen", "Copenhagen" "Lissabon", "Lisbon" "Ljubljana", "Luxemburg",
                  "Luxembourg" "Nikosia", "Nicosia", "Prag", "Prague", "Oslo", "Riga", "Sofia, Stockholm", "Tallinn",
                  "Valletta", "Vilnius", "Zagreb", "Barcelona", "Vatikan", "Vatican", "Schengen"]\


    _rusukr_me_cities = ["Iraq", "Irak", "Syria", "Syrien", "Libanon", "Lebanon", "Israel", "Jordan", "Jordanien",
                      "Ägypten", "Egypt", "Saudi", "Yemen", "Jemen", "Turkey", "Türkei",
                      "Qatar", "Katar", "Bahrain", "Iran", "Afghanistan", "Pakistan"]


    _syrirq_cities = ["Damaskus", "Damascus", "Raqqa", "Al-Raqqa", "Ar-Raqqa", "Alraqqa", "Arraqqa", "Homs", "Aleppo",
                      "Idlib", "Mossul", "Baghdad", "Basra", "Erbil"]
    _me_cities = ["Tel Aviv", "Jerusalem", "Beirut", "Amman", "Tehran", "Teheran",
                  "Kairo", "Cairo", "Alexandria", "Tripoli", "Tripolis",
                  "Doha", "Dubai", "Abu Dhabi",
                  "Kandahar", "Kabul", "Mazar-e Scharif", "Mazar-e Sharif", "Kundus",
                  "Ankara", "Istanbul"]
    _rusukr_countries = ["Ukraine", "Russland", "Belarus", "Weißrussland", "Russia"]
    _rusukr_cities = ["Moskau", "Moscow", "Kiew", "Kiev", "Minsk",
                      "Charkiw", "Charkow", "Kharkiw", "Kharkow", "Charkov", "Charkiv", "Kharkov", "Kharkiv",
                      "Odesa", "Odessa", "Dnipro", "Krim", "Crimea", "Donezk", "Donetsk", "Donesk",
                      "Lwiw", "Lviv", "Mariupol", "Donbas", "Donbass", "Luhansk",
                      "Saporischschja", "Zaporishshia", "Saporischtschja", "Saporizhzhja", "Zaporizhzhia",
                      "Zaporizhzhja", "Zaporozhye",
                      "Cherson", "Kherson"]

    _eu_countries = ["Belgien", "Bulgarien", "Dänemark", "Estland", "Finnland", "Frankreich", "Griechenland",
                  "Italien", "Kroatien", "Lettland", "Litauen", "Malta", "Niederlande", "Polen",
                  "Portugal", "Rumänien", "Schweden", "Slowakei", "Slowenien", "Spanien",
                  "Tschechien", "Ungarn", "Zypern", "Irland"] \
                    + ["Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech", "Denmark", "Estonia", "Finland", "France",
                    "Greece",
                    "Hungary", "Italy", "Latvia", "Ireland", "Italy", "Lithuania", "Malta", "Netherlands",
                    "Poland",
                    "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"]
    _eu_regions = ["Katalonien", "Catalunya", "Catalonia"]


    _oth_cities = _eu_cities + _syrirq_cities + _me_cities + _rusukr_cities + _rusukr_me_cities + _eu_countries + _eu_regions

    _oth_geographie = ["Mallorca", "Ibiza"]+\
                 ["Euphrat", "Tigris", "Nil", "Nile", "Jordan", "Suez",
                    "Hindukush", "Hindukusch", "Hindu Kusch", "Hindu Kush"]\
                   +["Asien", "Asia", "Afrika", "Africa", "Amerika", "America", "Australia", "Australien", "Antarktis",
                   "Arctic", "Arktis", "Antarctica", "Ozeanien", "Oceania", "Europa", "Europe"]


    O_NE_REGs = [word.lower() for word in _oth_cities + _oth_geographie]

    _oth_regions_abbr = ["EU"]
    O_NE_REGs_abbr = [word.lower() for word in _oth_regions_abbr]

    O_REG_demonym_verisons = ["Röm","Syr","Liban", "Leban","Ägypt","Türk","Turk","Afghan","Ukrain", "Russ", "Weißruss","Belgi", "Bulgar","Däne","Dan","Est","Finn","Franz","Frank","French","Greek","Griech","Ital","Kroat","Croat","Lett","Latv","Lithuan","Litau","Malt","Pol","Portug","Rumän","Roman","Sched","Swed","Slovak","Slowak","Slowen","Sloven", "Spani","Spani","Tschech","Czech","Ungar","Zypri","Cypri","Ir","Catalan","Katalan","Asiat", "Australi", "Europ"]




    # GERMAN ORGS
    _d_parteien = ["SPD", "CDU", "AfD", "FDP", "Linke", "die Linke", "Linken", "Grünen", "die Grünen", "Grüne", "Union"]
    _ö_parteien = ["ÖVP", "SPÖ", "FPÖ"]
    _d_orgs_ = ["EWR", "EuGH"]

    _deu_parteien = _d_parteien + _ö_parteien + _d_orgs_

    D_NE_ORGs = [word.lower() for word in _deu_parteien]

    # ENGLISH ORGANIZATIONS
    _e_orgs_ = ["NATO", "UN", "UNO", "UNESCO", "UNHCR","UNICEF"] + ["NSA", "FBI", "CSI", "CIA"] + ["Tories", "Labour",
                                                                                          "Republicans", "Democrats"]
    _eng_parteien = _e_orgs_

    E_NE_ORGs = [word.lower() for word in _eng_parteien]

    # OTHER ORGANIZATIONS
    _o_orgs_ = ["EU", "ditib"]
    _me_parteien = ["PKK", "HDP", "AKP", "CHP", "MHP",
                    "ISIS", "ISIL", "IS", "al-Qaida", "al Kaida", "al-Qaeda", "alqaida", "alqaeda", "al-Nusra",
                    "Peschmerga", "Peshmerga", "Taliban", "FSA",
                    "Hamas", "Hezbollah", "Hisbollah", "Hisbolla", "IRGC",
                    "Likud", "JA", "KL", "VL"]
    _eu_parteien = ["PP", "VOX", "PSOE", "UP", "Cs", "PiS", "KO", "PO",
                    "Fidesz", "EM", "Jobbik", "MSZP"]

    _oth_parteien = _o_orgs_ + _me_parteien + _eu_parteien

    O_NE_ORGs = [word.lower() for word in _oth_parteien]

    # GERMAN VIPs:
    _d_politiker = ["Hitler", "Merkel", "Angela Merkel", "Scholz", "Olaf Scholz",
                    "von der Leyen", "Ursula", "Ursula von der Leyen", "Bärbock", "Annalena", "Annalena Bärbock",
                    "Christian Lindner", "Lindner",
                    "Annegret Kramp-Karrenbauer", "Annegret", "Kramp-Karrenbauer", "Schmidt", "Helmut Schmidt",
                    "Willy Brandt",
                    "Brandt", "Kohl", "Helmut Kohl", "Schröder", "Gerhard Schröder"]
    _ö_politiker = ["Kurz", "Sebastian Kurz"]
    _d_sportler = ["Bastian Schweinsteiger", "Schweinsteiger", "Thomas Müller", "Müller", "Manuel Neuer", "Neuer",
                   "Götze", "Mario Götze",
                   "Rüdiger", "Philipp Lahm", "Lahm", "Per Merteskacker", "Mertesacker", "Klose", "Joachim", "Löw",
                   "Jogi", "Hansi Flick", "Flick"]
    _d_stars = ["Joko Winterscheid", "Joko", "Stefan Raab", "Raab", "Klaas Heufer-Umlauf", "Klaas"]

    _deu_vips = _d_politiker + _ö_politiker + _d_sportler + _d_stars

    D_NE_VIPs = [word.lower() for word in _deu_vips]

    NE_MEASURE = ["Pound", "Pfund", "Mile", "Meile", "Yard", "Foot", "Dollar", "Euro", "Lira", "Mark", "Pfennig",
                  "Penny", "Thaler", "Taler", "second", "Sekunde",
                  "Gramme", "Gram", "Gramm", "bit", "byte", "Dinar", "Watt", "Volt", "Pascal", "Joule", "Meter",
                  "Minute"]

    # ENGLISHE VIPs
    _us_politiker = ["Barack Obama", "Obama", "Joe Biden", "Biden", "Trump", "Donald Trump", "Hillary Clinton",
                     "Clinton",
                     "Washington", "George Washington", "Thomas Jefferson", "Jefferson", "Abraham Lincoln", "Lincoln",
                     "Roosevelt", "Eisenhower", "Kennedy",
                     "John F. Kennedy", "John Kennedy", "JFK", "Nixon", "Richard Nixon", "Reagan", "Ronald Reagan",
                     "Bill Clinton",
                     "George Bush", "Bush", "George W. Bush"]
    _uk_politiker = ["Boris Johnson", "Johnson", "Elisabeth"]
    _us_unternehmer = ["Elon Musk", "Musk", "Marc Zuckerberg", "Zuckerberg", "Bill Gates", "Gates", "Steve Jobs",
                       "Jobs"]

    _eng_vips = _us_politiker + _uk_politiker + _us_unternehmer

    E_NE_VIPs = [word.lower() for word in _eng_vips]

    # OTHER VIPs
    _eu_politiker = ["Hollande", "Francois Hollande", "Macron", "Emmanuel Macron",
                     "Orban", "Viktor Orban", "Sanchez", "Pedro Sanchez"]
    _me_politiker = ["Assad", "Gaddafi", "Erdogan",
                     "Netanjahu", "Benjamin Netanjahu",
                     "bin Laden", "Osama bin Laden", "Osama", "al-Baghdadi"]
    _rus_politiker = ["Stalin", "Gorbatschow", "Putin", "Vladimir Putin", "Vladimir", "Wladimir", "Wladimir Putin",
                      "Selenskyj", "Selensky", "Klitschko"]

    _oth_politiker = _eu_politiker + _me_politiker + _rus_politiker

    O_NE_VIPs = [word.lower() for word in _oth_politiker]

    # GERMAN PRESS
    _d_presse = ["ARD", "ZDF", "RTL", "Pro7", "Kabel1", "VOX", "ARTE", "3Sat",
                 "Phönix", "ORF", "SRF",
                 "WDR", "NDR", "RBB", "HR", "SR", "SWR", "MDR", "BR",
                 "Bild", "Welt", "Springer", "Spiegel", "Tagesschau", "Heute",
                 "FAZ", "TAZ", "dpa"]

    D_NE_PRESS = [word.lower() for word in _d_presse]

    _e_presse = ["CNN", "BBC", "MTV", "FOX"]

    E_NE_PRESS = [word.lower() for word in _e_presse]

    _o_presse = ["aljazeera", "al-Jazeera"]
    O_NE_PRESS = [word.lower() for word in _o_presse]

    # GERMAN COMPANIES
    _d_sonstige = ["SAP", "Siemens", "Telekom", "Allianz", "Bayer", "BASF", "Adidas", "Lufthansa", "Shell", "Fresenius",
                   "TUI", "Bertelsmann", "REW"]
    _d_autos = ["VW", "Volkswagen", "Mercedes", "Merzedes", "Benz", "BMW", "Audi", "Open", "Porsche"]
    _d_essen = ["Lidl", "Aldi", "Netto", "Penny", "Rewe", "Real", "Globus", "Edeka", "Wasgau"]
    _d_einkaufen = ["Kik", "Kaufland", "Kaufhof", "Karstadt"]
    _deu_comps = _d_autos + _d_essen + _d_sonstige + _d_einkaufen

    D_NE_COMPs = [word.lower() for word in _deu_comps]

    # ENGLISH COMPANIES
    _us_techcomps = ["Apple", "Google", "Android", "Microsoft", "Amazon",
                     "Facebook", "Twitter", "Instagram", "ebay", "YouTube",
                     "Linux", "Windows", "iOS", "Mac"] + \
                    ["Silicon Valley"]
    _us_autos = ["Cadillac", "Ford", "Jeep"]
    _eng_comps = _us_techcomps
    E_NE_COMPs = [word.lower() for word in _eng_comps]

    _o_comps = ["Samsung", "Nokia", "Toyota", "Kia", "Dacia", "Skoda"] + \
               ["Tesla", "Ferrari"]

    O_NE_COMPs = [word.lower() for word in _o_comps]


class CultureTerms:
    """contains lists of Cultural Terms"""

    _d_dish = ["bratwurst", "schnitzel", "brezel", "knödel", "reibekuchen", "kartoffelpuffer",
               "maultaschen", "spätzle", "sauerkraut", "kraut", "rouladen", "rollmops", "brot",
               "stollen", "plätzchen", "apfelstrudel", "sauerbraten", "wurst", "currywurst"]
    _d_holiday = []
    D_CULT = [word.lower() for word in _d_holiday + _d_dish]

    e_food = ["burger", "fish n chips", "fish 'n' chips", "fish `n' chips", "fish 'n chips", "hamburger",
              "cheeseburger"]
    e_holidays = ["Halloween", "Thanksgiving"]
    e_technology_terms = ["hardware", "software"]
    E_CULT = [word.lower() for word in e_holidays + e_food + e_technology_terms]

    _o_dish = ["döner", "doner", "kebab", "falafel", "gyros", "pizza", "crepe", "baguette", "sushi", "croissant"]

    _o_politics = ["caliphate", "Kalifat", "Califat", "Sultanate", "sultanat", "Emirat", "emirate", "sultan", "emir",
                   "caliphe", "calif", "Kaliph", "kalif"]
    _o_religion_isl = ["muezzin", "Eid", "Allah", "Quran", "Koran", "Jihad", "Mujaheed", "Mujahid", "Dschihad",
                       "Mudschaheed", "Mudschahed", "Mudschahid", "Kafir", "Islam", "Muslim", "Hadith"]
    _o_religion_chr = ["Bibel"]
    _o_religion_jud = ["Torah", "Talmud", "Hanukka", "Bar Mitzwa", "Bat Mitzwa", "Bar Mitswa", "Bat Mitswa",
                       "Bar Mitsva", "Bat Mitsva", "Bar Mitzva", "Bat Mitzva",
                       "Purim", "Pessach", "Passah", "Passach", "Pessah"]
    _o_religion = _o_religion_isl + _o_religion_chr + _o_religion_jud

    _o_weather = ["Tsunami", "Typhoon", "Taifun"]

    O_CULT = [word.lower() for word in _o_dish + _o_politics + _o_religion + _o_weather]



class OtherLists:
    Interj_Words = ["lmao", "rofl", "lol", "lel", "yolo", "wtf", "ftw"]

    URL_INFIX = ["@",".co."]
    URL_PREF = ["/r/", "/u/", "u/", "r/", "http", "www", "#"]
    URL_SUFF = [".de", ".com", ".eu", ".at", ".ch"]

    EMOTICON_LIST = [":‑)", ":)", ":-]", ":]", ":->", ":>", "8-)", "8)", ":-}", ":}", ":o)", ":c)", ":^)", "=]", "=)",
                     ":‑D", ":D", "8‑D", "8D", "=D", "=3", "B^D", "c:", "C:",
                     "x‑D", "xD", "X‑D", "XD",
                     ":-))",
                     ":‑(", ":(", ":‑c", ":c", ":‑<", ":<", ":‑[", ":[", ":-||", ":{", ":@", ":(", ";(",
                     ":'‑(", ":'(", ":=(", ":'‑)", ":')", ":\"D", ">:(", ">:[",
                     "D‑':", "D:<", "D:", "D8", "D;", "D=", "DX", ":‑O", ":O", ":‑o", ":o", ":-0", ":0", "8‑0", ">:O",
                     "=O", "=o", "=0",
                     ":-3", ":3", "=3", "x3", "X3", ">:3", ":-*", ":*", ":×",
                     ";‑)", ";)", "*-)", "*)", ";‑]", ";]", ";^)", ";>", ":‑,", ";D", ";3",
                     ":‑P", ":P", "X‑P", "XP", "x‑p", "xp", ":‑p", ":p", ":‑Þ", ":Þ", ":‑þ", ":þ", ":‑b", ":b", "d:",
                     "=p", ">:P",
                     ":-/", ":/", ":‑.", ">:\\", ">:/", ":\\", "=/", "=\\", ":L", "=L", ":S",
                     ":‑|", ":|", ":$", "://)", "://3", ":‑X", ":X", ":‑#", ":#", ":‑&", ":&",
                     "O:‑)", "O:)", "0:‑3", "0:3", "0:‑)", "0:)", "0;^)",
                     ">:‑)", ">:)", "}:‑)", "}:)", "3:‑)", "3:)", ">;‑)", ">;)", ">:3", ";3",
                     "|;‑)", "|‑O", "B-)",
                     ":‑J", "#‑)", "%‑)", "%)", "</3", "<\\3", "<3",
                     "\o/", "*\\0/*",
                     "v.v", "._.", "._.;",
                     "QQ", "qq", "Qq", "Q.Q", "q.q", "Q.q", "Q_Q", "q_q", "Q_q",
                     "X_X", "x_x", "+_+", "X_x", "x_X",
                     "<_<", ">_>", "<.<", ">.>",
                     "O_O", "o_o", "O-O", "o‑o", "O_o", "o_O",
                     ">.<", ">_<"
                     ]


class FreqLists:
    # wikipedia: https://de.wikipedia.org/wiki/Liste_der_häufigsten_Wörter_der_deutschen_Sprache
    DE_WORD_LIST = ['die', 'der', 'und', 'in', 'zu', 'den', 'das', 'nicht', 'von', 'sie', 'ist', 'des', 'sich', 'mit',
                    'dem', 'dass', 'er', 'es', 'ein', 'ich', 'auf', 'so', 'eine', 'auch', 'als', 'an', 'nach', 'wie',
                    'im',
                    'für', 'man', 'aber', 'aus', 'durch', 'wenn', 'nur', 'war', 'noch', 'werden', 'bei', 'hat', 'wir',
                    'was', 'wird', 'sein', 'einen', 'welche', 'sind', 'oder', 'zur', 'um', 'haben', 'einer', 'mir',
                    'über',
                    'ihm', 'diese', 'einem', 'ihr', 'uns', 'da', 'zum', 'kann', 'doch', 'vor', 'dieser', 'mich', 'ihn',
                    'du', 'hatte', 'seine', 'mehr', 'am', 'denn', 'nun', 'unter', 'sehr', 'selbst', 'schon', 'hier',
                    'bis',
                    'habe', 'ihre', 'dann', 'ihnen', 'seiner', 'alle', 'wieder', 'meine', 'Zeit', 'gegen', 'vom',
                    'ganz',
                    'einzelnen', 'wo', 'muss', 'ohne', 'eines', 'können', 'sei', 'ja', 'wurde', 'jetzt', 'immer',
                    'seinen',
                    'wohl', 'dieses', 'ihren', 'würde', 'diesen', 'sondern', 'weil', 'welcher', 'nichts', 'diesem',
                    'alles',
                    'waren', 'will', 'Herr', 'viel', 'mein', 'also', 'soll', 'worden', 'lassen', 'dies', 'machen',
                    'ihrer',
                    'weiter', 'Leben', 'recht', 'etwas', 'keine', 'seinem', 'ob', 'dir', 'allen', 'großen', 'Jahre',
                    'Weise', 'müssen', 'welches', 'wäre', 'erst', 'einmal', 'Mann', 'hätte', 'zwei', 'dich', 'allein',
                    'Herren', 'während', 'Paragraph', 'anders', 'Liebe', 'kein', 'damit', 'gar', 'Hand', 'Herrn',
                    'euch',
                    'sollte', 'konnte', 'ersten', 'deren', 'zwischen', 'wollen', 'denen', 'dessen', 'sagen', 'bin',
                    'Menschen', 'gut', 'darauf', 'wurden', 'weiß', 'gewesen', 'Seite', 'bald', 'weit', 'große',
                    'solche',
                    'hatten', 'eben', 'andern', 'beiden', 'macht', 'sehen', 'ganze', 'anderen', 'lange', 'wer', 'ihrem',
                    'zwar', 'gemacht', 'dort', 'kommen', 'Welt', 'heute', 'Frau', 'werde', 'derselben', 'ganzen',
                    'deutschen', 'lässt', 'vielleicht', 'meiner']
    # 5050 most freuqent English words, taken from the one billion word Corpus of Contemporary American English
    EN_WORD_LIST = ['the', 'be', 'and', 'a', 'of', 'to', 'in', 'i', 'you', 'it', 'have', 'to', 'that', 'for', 'do',
                    'he', 'with', 'on', 'this', "n't", 'we', 'that', 'not', 'but', 'they', 'say', 'at', 'what', 'his',
                    'from', 'go', 'or', 'by', 'get', 'she', 'my', 'can', 'as', 'know', 'if', 'me', 'your', 'all',
                    'who', 'about', 'their', 'will', 'so', 'would', 'make', 'just', 'up', 'think', 'time', 'there',
                    'see', 'her', 'as', 'out', 'one', 'come', 'people', 'take', 'year', 'him', 'them', 'some', 'want',
                    'how', 'when', 'which', 'now', 'like', 'other', 'could', 'our', 'into', 'here', 'then', 'than',
                    'look', 'way', 'more', 'these', 'no', 'thing', 'well', 'because', 'also', 'two', 'use', 'tell',
                    'good', 'first', 'man', 'day', 'find', 'give', 'more', 'new', 'one', 'us', 'any', 'those', 'very',
                    'her', 'need', 'back', 'there', 'should', 'even', 'only', 'many', 'really', 'work', 'life', 'why',
                    'right', 'down', 'on', 'try', 'let', 'something', 'too', 'call', 'woman', 'may', 'still',
                    'through', 'mean', 'after', 'never', 'no', 'world', 'in', 'feel', 'yeah', 'great', 'last', 'child',
                    'oh', 'over', 'ask', 'when', 'as', 'school', 'state', 'much', 'talk', 'out', 'keep', 'leave',
                    'put', 'like', 'help', 'big', 'where', 'same', 'all', 'own', 'while', 'start', 'three', 'high',
                    'every', 'another', 'become', 'most', 'between', 'happen', 'family', 'over', 'president', 'old',
                    'yes', 'house', 'show', 'again', 'student', 'so', 'seem', 'might', 'part', 'hear', 'its', 'place',
                    'problem', 'where', 'believe', 'country', 'always', 'week', 'point', 'hand', 'off', 'play', 'turn',
                    'few', 'group', 'such', 'against', 'run', 'guy', 'about', 'case', 'question', 'work', 'night',
                    'live', 'game', 'number', 'write', 'bring', 'without', 'money', 'lot', 'most', 'book', 'system',
                    'government', 'next', 'city', 'company', 'story', 'today', 'job', 'move', 'must', 'bad', 'friend',
                    'during', 'begin', 'love', 'each', 'hold', 'different', 'american', 'little', 'before', 'ever',
                    'word', 'fact', 'right', 'read', 'anything', 'nothing', 'sure', 'small', 'month', 'program',
                    'maybe', 'right', 'under', 'business', 'home', 'kind', 'stop', 'pay', 'study', 'since', 'issue',
                    'name', 'idea', 'room', 'percent', 'far', 'away', 'law', 'actually', 'large', 'though', 'provide',
                    'lose', 'power', 'kid', 'war', 'understand', 'head', 'mother', 'real', 'best', 'team', 'eye',
                    'long', 'long', 'side', 'water', 'young', 'wait', 'okay', 'both', 'yet', 'after', 'meet',
                    'service', 'area', 'important', 'person', 'hey', 'thank', 'much', 'someone', 'end', 'change',
                    'however', 'only', 'around', 'hour', 'everything', 'national', 'four', 'line', 'girl', 'around',
                    'watch', 'until', 'father', 'sit', 'create', 'information', 'car', 'learn', 'least', 'already',
                    'kill', 'minute', 'party', 'include', 'stand', 'together', 'back', 'follow', 'health', 'remember',
                    'often', 'reason', 'speak', 'ago', 'set', 'black', 'member', 'community', 'once', 'social', 'news',
                    'allow', 'win', 'body', 'lead', 'continue', 'whether', 'enough', 'spend', 'level', 'able',
                    'political', 'almost', 'boy', 'university', 'before', 'stay', 'add', 'later', 'change', 'five',
                    'probably', 'center', 'among', 'face', 'public', 'die', 'food', 'else', 'history', 'buy', 'result',
                    'morning', 'off', 'parent', 'office', 'course', 'send', 'research', 'walk', 'door', 'white',
                    'several', 'court', 'home', 'grow', 'better', 'open', 'moment', 'including', 'consider', 'both',
                    'such', 'little', 'within', 'second', 'late', 'street', 'free', 'better', 'everyone', 'policy',
                    'table', 'sorry', 'care', 'low', 'human', 'please', 'hope', 'TRUE', 'process', 'teacher', 'data',
                    'offer', 'death', 'whole', 'experience', 'plan', 'easy', 'education', 'build', 'expect', 'fall',
                    'himself', 'age', 'hard', 'sense', 'across', 'show', 'early', 'college', 'music', 'appear', 'mind',
                    'class', 'police', 'use', 'effect', 'season', 'tax', 'heart', 'son', 'art', 'possible', 'serve',
                    'break', 'although', 'end', 'market', 'even', 'air', 'force', 'require', 'foot', 'up', 'listen',
                    'agree', 'according', 'anyone', 'baby', 'wrong', 'love', 'cut', 'decide', 'republican', 'full',
                    'behind', 'pass', 'interest', 'sometimes', 'security', 'eat', 'report', 'control', 'rate', 'local',
                    'suggest', 'report', 'nation', 'sell', 'action', 'support', 'wife', 'decision', 'receive', 'value',
                    'base', 'pick', 'phone', 'thanks', 'event', 'drive', 'strong', 'reach', 'remain', 'explain',
                    'site', 'hit', 'pull', 'church', 'model', 'perhaps', 'relationship', 'six', 'fine', 'movie',
                    'field', 'raise', 'less', 'player', 'couple', 'million', 'themselves', 'record', 'especially',
                    'difference', 'light', 'development', 'federal', 'former', 'role', 'pretty', 'myself', 'view',
                    'price', 'effort', 'nice', 'quite', 'along', 'voice', 'finally', 'department', 'either', 'toward',
                    'leader', 'because', 'photo', 'wear', 'space', 'project', 'return', 'position', 'special',
                    'million', 'film', 'need', 'major', 'type', 'town', 'article', 'road', 'form', 'chance', 'drug',
                    'economic', 'situation', 'choose', 'practice', 'cause', 'happy', 'science', 'join', 'teach',
                    'early', 'develop', 'share', 'yourself', 'carry', 'clear', 'brother', 'matter', 'dead', 'image',
                    'star', 'cost', 'simply', 'post', 'society', 'picture', 'piece', 'paper', 'energy', 'personal',
                    'building', 'military', 'open', 'doctor', 'activity', 'exactly', 'american', 'media', 'miss',
                    'evidence', 'product', 'realize', 'save', 'arm', 'technology', 'catch', 'comment', 'look', 'term',
                    'color', 'cover', 'describe', 'guess', 'choice', 'source', 'mom', 'soon', 'director',
                    'international', 'rule', 'campaign', 'ground', 'election', 'face', 'uh', 'check', 'page', 'fight',
                    'itself', 'test', 'patient', 'produce', 'certain', 'whatever', 'half', 'video', 'support', 'throw',
                    'third', 'care', 'rest', 'recent', 'available', 'step', 'ready', 'opportunity', 'official', 'oil',
                    'call', 'organization', 'character', 'single', 'current', 'likely', 'county', 'future', 'dad',
                    'whose', 'less', 'shoot', 'industry', 'second', 'list', 'general', 'stuff', 'figure', 'attention',
                    'forget', 'risk', 'no', 'focus', 'short', 'fire', 'dog', 'red', 'hair', 'point', 'condition',
                    'wall', 'daughter', 'before', 'deal', 'author', 'truth', 'upon', 'husband', 'period', 'series',
                    'order', 'officer', 'close', 'land', 'note', 'computer', 'thought', 'economy', 'goal', 'bank',
                    'behavior', 'sound', 'deal', 'certainly', 'nearly', 'increase', 'act', 'north', 'well', 'blood',
                    'culture', 'medical', 'ok', 'everybody', 'top', 'difficult', 'close', 'language', 'window',
                    'response', 'population', 'lie', 'tree', 'park', 'worker', 'draw', 'plan', 'drop', 'push', 'earth',
                    'cause', 'per', 'private', 'tonight', 'race', 'than', 'letter', 'other', 'gun', 'simple', 'course',
                    'wonder', 'involve', 'hell', 'poor', 'each', 'answer', 'nature', 'administration', 'common', 'no',
                    'hard', 'message', 'song', 'enjoy', 'similar', 'congress', 'attack', 'past', 'hot', 'seek',
                    'amount', 'analysis', 'store', 'defense', 'bill', 'like', 'cell', 'away', 'performance',
                    'hospital', 'bed', 'board', 'protect', 'century', 'summer', 'material', 'individual', 'recently',
                    'example', 'represent', 'fill', 'state', 'place', 'animal', 'fail', 'factor', 'natural', 'sir',
                    'agency', 'usually', 'significant', 'help', 'ability', 'mile', 'statement', 'entire', 'democrat',
                    'floor', 'serious', 'career', 'dollar', 'vote', 'sex', 'compare', 'south', 'forward', 'subject',
                    'financial', 'identify', 'beautiful', 'decade', 'bit', 'reduce', 'sister', 'quality', 'quickly',
                    'act', 'press', 'worry', 'accept', 'enter', 'mention', 'sound', 'thus', 'plant', 'movement',
                    'scene', 'section', 'treatment', 'wish', 'benefit', 'interesting', 'west', 'candidate', 'approach',
                    'determine', 'resource', 'claim', 'answer', 'prove', 'sort', 'enough', 'size', 'somebody',
                    'knowledge', 'rather', 'hang', 'sport', 'tv', 'loss', 'argue', 'left', 'note', 'meeting', 'skill',
                    'card', 'feeling', 'despite', 'degree', 'crime', 'that', 'sign', 'occur', 'imagine', 'vote',
                    'near', 'king', 'box', 'present', 'figure', 'seven', 'foreign', 'laugh', 'disease', 'lady',
                    'beyond', 'discuss', 'finish', 'design', 'concern', 'ball', 'east', 'recognize', 'apply',
                    'prepare', 'network', 'huge', 'success', 'district', 'cup', 'name', 'physical', 'growth', 'rise',
                    'hi', 'standard', 'force', 'sign', 'fan', 'theory', 'staff', 'hurt', 'legal', 'september', 'set',
                    'outside', 'et', 'strategy', 'clearly', 'property', 'lay', 'final', 'authority', 'perfect',
                    'method', 'region', 'since', 'impact', 'indicate', 'safe', 'committee', 'supposed', 'dream',
                    'training', 'shit', 'central', 'option', 'eight', 'particularly', 'completely', 'opinion', 'main',
                    'ten', 'interview', 'exist', 'remove', 'dark', 'play', 'union', 'professor', 'pressure', 'purpose',
                    'stage', 'blue', 'herself', 'sun', 'pain', 'artist', 'employee', 'avoid', 'account', 'release',
                    'fund', 'environment', 'treat', 'specific', 'version', 'shot', 'hate', 'reality', 'visit', 'club',
                    'justice', 'river', 'brain', 'memory', 'rock', 'talk', 'camera', 'global', 'various', 'arrive',
                    'notice', 'bit', 'detail', 'challenge', 'argument', 'lot', 'nobody', 'weapon', 'best', 'station',
                    'island', 'absolutely', 'instead', 'discussion', 'instead', 'affect', 'design', 'little', 'anyway',
                    'respond', 'control', 'trouble', 'conversation', 'manage', 'close', 'date', 'public', 'army',
                    'top', 'post', 'charge', 'seat', 'assume', 'writer', 'perform', 'credit', 'green', 'marriage',
                    'operation', 'indeed', 'sleep', 'necessary', 'reveal', 'agent', 'access', 'bar', 'debate', 'leg',
                    'contain', 'beat', 'cool', 'democratic', 'cold', 'glass', 'improve', 'adult', 'trade', 'religious',
                    'head', 'review', 'kind', 'address', 'association', 'measure', 'stock', 'gas', 'deep', 'lawyer',
                    'production', 'relate', 'middle', 'management', 'original', 'victim', 'cancer', 'speech',
                    'particular', 'trial', 'none', 'item', 'weight', 'tomorrow', 'step', 'positive', 'form', 'citizen',
                    'study', 'trip', 'establish', 'executive', 'politics', 'stick', 'customer', 'manager', 'rather',
                    'publish', 'popular', 'sing', 'ahead', 'conference', 'total', 'discover', 'fast', 'base',
                    'direction', 'sunday', 'maintain', 'past', 'majority', 'peace', 'dinner', 'partner', 'user',
                    'above', 'fly', 'bag', 'therefore', 'rich', 'individual', 'tough', 'owner', 'shall', 'inside',
                    'voter', 'tool', 'june', 'far', 'may', 'mountain', 'range', 'coach', 'fear', 'friday', 'attorney',
                    'unless', 'nor', 'expert', 'structure', 'budget', 'insurance', 'text', 'freedom', 'crazy',
                    'reader', 'style', 'through', 'march', 'machine', 'november', 'generation', 'income', 'born',
                    'admit', 'hello', 'onto', 'sea', 'okay', 'mouth', 'throughout', 'own', 'test', 'web', 'shake',
                    'threat', 'solution', 'shut', 'down', 'travel', 'scientist', 'hide', 'obviously', 'refer', 'alone',
                    'drink', 'investigation', 'senator', 'unit', 'photograph', 'july', 'television', 'key', 'sexual',
                    'radio', 'prevent', 'once', 'modern', 'senate', 'violence', 'touch', 'feature', 'audience',
                    'evening', 'whom', 'front', 'hall', 'task', 'score', 'skin', 'suffer', 'wide', 'spring',
                    'experience', 'civil', 'safety', 'weekend', 'while', 'worth', 'title', 'heat', 'normal', 'hope',
                    'yard', 'finger', 'tend', 'mission', 'eventually', 'participant', 'hotel', 'judge', 'pattern',
                    'break', 'institution', 'faith', 'professional', 'reflect', 'folk', 'surface', 'fall', 'client',
                    'edge', 'traditional', 'council', 'device', 'firm', 'environmental', 'responsibility', 'chair',
                    'internet', 'october', 'by', 'funny', 'immediately', 'investment', 'ship', 'effective', 'previous',
                    'content', 'consumer', 'element', 'nuclear', 'spirit', 'directly', 'afraid', 'define', 'handle',
                    'track', 'run', 'wind', 'lack', 'cost', 'announce', 'journal', 'heavy', 'ice', 'collection',
                    'feed', 'soldier', 'just', 'governor', 'fish', 'shoulder', 'cultural', 'successful', 'fair',
                    'trust', 'suddenly', 'future', 'interested', 'deliver', 'saturday', 'editor', 'fresh', 'anybody',
                    'destroy', 'claim', 'critical', 'agreement', 'powerful', 'researcher', 'concept', 'willing',
                    'band', 'marry', 'promise', 'easily', 'restaurant', 'league', 'senior', 'capital', 'anymore',
                    'april', 'potential', 'etc', 'quick', 'magazine', 'status', 'attend', 'replace', 'due', 'hill',
                    'kitchen', 'achieve', 'screen', 'generally', 'mistake', 'along', 'strike', 'battle', 'spot',
                    'basic', 'very', 'corner', 'target', 'driver', 'beginning', 'religion', 'crisis', 'count',
                    'museum', 'engage', 'communication', 'murder', 'blow', 'object', 'express', 'huh', 'encourage',
                    'matter', 'blog', 'smile', 'return', 'belief', 'block', 'debt', 'fire', 'labor', 'understanding',
                    'neighborhood', 'contract', 'middle', 'species', 'additional', 'sample', 'involved', 'inside',
                    'mostly', 'path', 'concerned', 'apple', 'conduct', 'god', 'wonderful', 'library', 'prison', 'hole',
                    'attempt', 'complete', 'code', 'sales', 'gift', 'refuse', 'increase', 'garden', 'introduce',
                    'roll', 'christian', 'definitely', 'like', 'lake', 'turn', 'sure', 'earn', 'plane', 'vehicle',
                    'examine', 'application', 'thousand', 'coffee', 'gain', 'result', 'file', 'billion', 'reform',
                    'ignore', 'welcome', 'gold', 'jump', 'planet', 'location', 'bird', 'amazing', 'principle',
                    'promote', 'search', 'nine', 'alive', 'possibility', 'sky', 'otherwise', 'remind', 'healthy',
                    'fit', 'horse', 'advantage', 'commercial', 'steal', 'basis', 'context', 'highly', 'christmas',
                    'strength', 'move', 'monday', 'mean', 'alone', 'beach', 'survey', 'writing', 'master', 'cry',
                    'scale', 'resident', 'football', 'sweet', 'failure', 'reporter', 'commit', 'fight', 'one',
                    'associate', 'vision', 'function', 'truly', 'sick', 'average', 'human', 'stupid', 'will',
                    'chinese', 'connection', 'camp', 'stone', 'hundred', 'key', 'truck', 'afternoon', 'responsible',
                    'secretary', 'apparently', 'smart', 'southern', 'totally', 'western', 'collect', 'conflict',
                    'burn', 'learning', 'wake', 'contribute', 'ride', 'british', 'following', 'order', 'share',
                    'newspaper', 'foundation', 'variety', 'perspective', 'document', 'presence', 'stare', 'lesson',
                    'limit', 'appreciate', 'complete', 'observe', 'currently', 'hundred', 'fun', 'crowd', 'attack',
                    'apartment', 'survive', 'guest', 'soul', 'protection', 'intelligence', 'yesterday', 'somewhere',
                    'border', 'reading', 'terms', 'leadership', 'present', 'chief', 'attitude', 'start', 'um', 'deny',
                    'website', 'seriously', 'actual', 'recall', 'fix', 'negative', 'connect', 'distance', 'regular',
                    'climate', 'relation', 'flight', 'dangerous', 'boat', 'aspect', 'grab', 'until', 'favorite',
                    'like', 'january', 'independent', 'volume', 'am', 'lots', 'front', 'online', 'theater', 'speed',
                    'aware', 'identity', 'demand', 'extra', 'charge', 'guard', 'demonstrate', 'fully', 'tuesday',
                    'facility', 'farm', 'mind', 'fun', 'thousand', 'august', 'hire', 'light', 'link', 'shoe',
                    'institute', 'below', 'living', 'european', 'quarter', 'basically', 'forest', 'multiple', 'poll',
                    'wild', 'measure', 'twice', 'cross', 'background', 'settle', 'winter', 'focus', 'presidential',
                    'operate', 'fuck', 'view', 'daily', 'shop', 'above', 'division', 'slowly', 'advice', 'reaction',
                    'injury', 'it', 'ticket', 'grade', 'wow', 'birth', 'painting', 'outcome', 'enemy', 'damage',
                    'being', 'storm', 'shape', 'bowl', 'commission', 'captain', 'ear', 'troop', 'female', 'wood',
                    'warm', 'clean', 'lead', 'minister', 'neighbor', 'tiny', 'mental', 'software', 'glad', 'finding',
                    'lord', 'drive', 'temperature', 'quiet', 'spread', 'bright', 'cut', 'influence', 'kick', 'annual',
                    'procedure', 'respect', 'wave', 'tradition', 'threaten', 'primary', 'strange', 'actor', 'blame',
                    'active', 'cat', 'depend', 'bus', 'clothes', 'affair', 'contact', 'category', 'topic', 'victory',
                    'direct', 'towards', 'map', 'egg', 'ensure', 'general', 'expression', 'past', 'session',
                    'competition', 'possibly', 'technique', 'mine', 'average', 'intend', 'impossible', 'moral',
                    'academic', 'wine', 'approach', 'somehow', 'gather', 'scientific', 'african', 'cook',
                    'participate', 'gay', 'appropriate', 'youth', 'dress', 'straight', 'weather', 'recommend',
                    'medicine', 'novel', 'obvious', 'thursday', 'exchange', 'explore', 'extend', 'bay', 'invite',
                    'tie', 'ah', 'belong', 'obtain', 'broad', 'conclusion', 'progress', 'surprise', 'assessment',
                    'smile', 'feature', 'cash', 'defend', 'pound', 'correct', 'married', 'pair', 'slightly', 'loan',
                    'village', 'half', 'suit', 'demand', 'historical', 'meaning', 'attempt', 'supply', 'lift',
                    'ourselves', 'honey', 'bone', 'consequence', 'unique', 'next', 'regulation', 'award', 'bottom',
                    'excuse', 'familiar', 'classroom', 'search', 'reference', 'emerge', 'long', 'lunch', 'judge', 'ad',
                    'desire', 'instruction', 'emergency', 'thinking', 'tour', 'french', 'combine', 'moon', 'sad',
                    'address', 'december', 'anywhere', 'chicken', 'fuel', 'train', 'abuse', 'construction',
                    'wednesday', 'link', 'deserve', 'famous', 'intervention', 'grand', 'visit', 'confirm', 'lucky',
                    'insist', 'coast', 'proud', 'cover', 'fourth', 'cop', 'angry', 'native', 'supreme', 'baseball',
                    'but', 'email', 'accident', 'front', 'duty', 'growing', 'struggle', 'revenue', 'expand', 'chief',
                    'launch', 'trend', 'ring', 'repeat', 'breath', 'inch', 'neck', 'core', 'terrible', 'billion',
                    'relatively', 'complex', 'press', 'miss', 'slow', 'soft', 'generate', 'extremely', 'last', 'drink',
                    'forever', 'corporate', 'deep', 'prefer', 'except', 'cheap', 'literature', 'direct', 'mayor',
                    'male', 'importance', 'record', 'danger', 'emotional', 'knee', 'ass', 'capture', 'traffic',
                    'fucking', 'outside', 'now', 'train', 'plate', 'equipment', 'select', 'file', 'studio',
                    'expensive', 'secret', 'engine', 'adopt', 'luck', 'via', 'pm', 'panel', 'hero', 'circle', 'critic',
                    'solve', 'busy', 'episode', 'back', 'check', 'requirement', 'politician', 'rain', 'colleague',
                    'disappear', 'beer', 'predict', 'exercise', 'tired', 'democracy', 'ultimately', 'setting', 'honor',
                    'works', 'unfortunately', 'theme', 'issue', 'male', 'clean', 'united', 'pool', 'educational',
                    'empty', 'comfortable', 'investigate', 'useful', 'pocket', 'digital', 'plenty', 'entirely', 'fear',
                    'afford', 'sugar', 'teaching', 'conservative', 'chairman', 'error', 'bridge', 'tall',
                    'specifically', 'flower', 'though', 'universe', 'live', 'acknowledge', 'limit', 'coverage', 'crew',
                    'locate', 'balance', 'equal', 'lip', 'lean', 'zone', 'wedding', 'copy', 'score', 'joke', 'used',
                    'clear', 'bear', 'meal', 'review', 'minority', 'sight', 'sleep', 'russian', 'dress', 'release',
                    'soviet', 'profit', 'challenge', 'careful', 'gender', 'tape', 'ocean', 'unidentified', 'host',
                    'grant', 'circumstance', 'late', 'boss', 'declare', 'domestic', 'tea', 'organize', 'english',
                    'neither', 'either', 'official', 'surround', 'manner', 'surprised', 'percentage', 'massive',
                    'cloud', 'winner', 'honest', 'standard', 'propose', 'rely', 'plus', 'sentence', 'request',
                    'appearance', 'regarding', 'excellent', 'criminal', 'salt', 'beauty', 'bottle', 'component',
                    'under', 'fee', 'jewish', 'yours', 'dry', 'dance', 'shirt', 'tip', 'plastic', 'indian', 'mark',
                    'tooth', 'meat', 'stress', 'illegal', 'significantly', 'february', 'constitution', 'definition',
                    'uncle', 'metal', 'album', 'self', 'suppose', 'investor', 'fruit', 'holy', 'desk', 'eastern',
                    'valley', 'largely', 'abortion', 'chapter', 'commitment', 'celebrate', 'arrest', 'dance', 'prime',
                    'urban', 'internal', 'bother', 'proposal', 'shift', 'capacity', 'guilty', 'warn', 'influence',
                    'weak', 'except', 'catholic', 'nose', 'variable', 'convention', 'jury', 'root', 'incident',
                    'climb', 'hearing', 'everywhere', 'payment', 'bear', 'conclude', 'scream', 'surgery', 'shadow',
                    'witness', 'increasingly', 'chest', 'amendment', 'paint', 'secret', 'complain', 'extent',
                    'pleasure', 'nod', 'holiday', 'super', 'talent', 'necessarily', 'liberal', 'expectation', 'ride',
                    'accuse', 'knock', 'previously', 'wing', 'corporation', 'sector', 'fat', 'experiment', 'match',
                    'thin', 'farmer', 'rare', 'english', 'confidence', 'bunch', 'bet', 'cite', 'northern', 'speaker',
                    'breast', 'contribution', 'leaf', 'creative', 'interaction', 'hat', 'doubt', 'promise', 'pursue',
                    'overall', 'nurse', 'question', 'long-term', 'gene', 'package', 'weird', 'difficulty', 'hardly',
                    'daddy', 'estimate', 'list', 'era', 'comment', 'aid', 'vs', 'invest', 'personally', 'notion',
                    'explanation', 'airport', 'chain', 'expose', 'lock', 'convince', 'channel', 'carefully', 'tear',
                    'estate', 'initial', 'offer', 'purchase', 'guide', 'forth', 'his', 'bond', 'birthday', 'travel',
                    'pray', 'improvement', 'ancient', 'ought', 'escape', 'trail', 'brown', 'fashion', 'length',
                    'sheet', 'funding', 'meanwhile', 'fault', 'barely', 'eliminate', 'motion', 'essential', 'apart',
                    'combination', 'limited', 'description', 'mix', 'snow', 'implement', 'pretty', 'proper', 'part',
                    'marketing', 'approve', 'other', 'bomb', 'slip', 'regional', 'lack', 'muscle', 'contact', 'rise',
                    'false', 'likely', 'creation', 'typically', 'spending', 'instrument', 'mass', 'far', 'thick',
                    'kiss', 'increased', 'inspire', 'separate', 'noise', 'yellow', 'aim', 'e-mail', 'cycle', 'signal',
                    'app', 'golden', 'reject', 'inform', 'perception', 'visitor', 'cast', 'contrast', 'judgment',
                    'mean', 'rest', 'representative', 'pass', 'regime', 'merely', 'producer', 'whoa', 'route', 'lie',
                    'typical', 'analyst', 'account', 'elect', 'smell', 'female', 'living', 'disability', 'comparison',
                    'hand', 'rating', 'campus', 'assess', 'solid', 'branch', 'mad', 'somewhat', 'gentleman',
                    'opposition', 'fast', 'suspect', 'land', 'hit', 'aside', 'athlete', 'opening', 'prayer',
                    'frequently', 'employ', 'basketball', 'existing', 'revolution', 'click', 'emotion', 'fuck',
                    'platform', 'behind', 'frame', 'appeal', 'quote', 'potential', 'struggle', 'brand', 'enable',
                    'legislation', 'addition', 'lab', 'oppose', 'row', 'immigration', 'asset', 'observation', 'online',
                    'taste', 'decline', 'attract', 'ha', 'for', 'household', 'separate', 'breathe', 'existence',
                    'mirror', 'pilot', 'stand', 'relief', 'milk', 'warning', 'heaven', 'flow', 'literally', 'quit',
                    'calorie', 'seed', 'vast', 'bike', 'german', 'employer', 'drag', 'technical', 'disaster',
                    'display', 'sale', 'bathroom', 'succeed', 'consistent', 'agenda', 'enforcement', 'diet', 'mark',
                    'silence', 'journalist', 'bible', 'queen', 'divide', 'expense', 'cream', 'exposure', 'priority',
                    'soil', 'angel', 'journey', 'trust', 'relevant', 'tank', 'cheese', 'schedule', 'bedroom', 'tone',
                    'selection', 'date', 'perfectly', 'wheel', 'gap', 'veteran', 'below', 'disagree', 'characteristic',
                    'protein', 'resolution', 'whole', 'regard', 'fewer', 'engineer', 'walk', 'dish', 'waste', 'print',
                    'depression', 'dude', 'fat', 'present', 'upper', 'wrap', 'ceo', 'visual', 'initiative', 'rush',
                    'gate', 'slow', 'whenever', 'entry', 'japanese', 'gray', 'assistance', 'height', 'compete', 'rule',
                    'due', 'essentially', 'benefit', 'phase', 'conservative', 'recover', 'criticism', 'faculty',
                    'achievement', 'alcohol', 'therapy', 'offense', 'touch', 'killer', 'personality', 'landscape',
                    'deeply', 'reasonable', 'soon', 'suck', 'transition', 'fairly', 'column', 'wash', 'button',
                    'opponent', 'pour', 'immigrant', 'first', 'distribution', 'golf', 'pregnant', 'unable',
                    'alternative', 'favorite', 'stop', 'violent', 'portion', 'acquire', 'suicide', 'stretch',
                    'deficit', 'symptom', 'solar', 'complaint', 'capable', 'analyze', 'scared', 'supporter', 'dig',
                    'twenty', 'pretend', 'philosophy', 'childhood', 'lower', 'well', 'outside', 'dark', 'wealth',
                    'welfare', 'poverty', 'prosecutor', 'spiritual', 'double', 'evaluate', 'mass', 'israeli', 'shift',
                    'reply', 'buck', 'display', 'knife', 'round', 'tech', 'detective', 'pack', 'disorder', 'creature',
                    'tear', 'closely', 'industrial', 'housing', 'watch', 'chip', 'regardless', 'numerous', 'tie',
                    'range', 'command', 'shooting', 'dozen', 'pop', 'layer', 'bread', 'exception', 'passion', 'block',
                    'highway', 'pure', 'commander', 'extreme', 'publication', 'vice', 'fellow', 'win', 'mystery',
                    'championship', 'install', 'tale', 'liberty', 'host', 'beneath', 'passenger', 'physician',
                    'graduate', 'sharp', 'substance', 'atmosphere', 'stir', 'muslim', 'passage', 'pepper', 'emphasize',
                    'cable', 'square', 'recipe', 'load', 'beside', 'roof', 'vegetable', 'accomplish', 'silent',
                    'habit', 'discovery', 'total', 'recovery', 'dna', 'gain', 'territory', 'girlfriend', 'consist',
                    'straight', 'surely', 'proof', 'nervous', 'immediate', 'parking', 'sin', 'unusual', 'rice',
                    'engineering', 'advance', 'interview', 'bury', 'still', 'cake', 'anonymous', 'flag',
                    'contemporary', 'good', 'jail', 'rural', 'match', 'coach', 'interpretation', 'wage', 'breakfast',
                    'severe', 'profile', 'saving', 'brief', 'adjust', 'reduction', 'constantly', 'assist', 'bitch',
                    'constant', 'permit', 'primarily', 'entertainment', 'shout', 'academy', 'teaspoon', 'dream',
                    'transfer', 'usual', 'ally', 'clinical', 'count', 'swear', 'avenue', 'priest', 'employment',
                    'waste', 'relax', 'owe', 'transform', 'grass', 'narrow', 'ethnic', 'scholar', 'edition', 'abandon',
                    'practical', 'infection', 'musical', 'suggestion', 'resistance', 'smoke', 'prince', 'illness',
                    'embrace', 'trade', 'republic', 'volunteer', 'target', 'general', 'evaluation', 'mine', 'opposite',
                    'awesome', 'switch', 'black', 'iraqi', 'iron', 'perceive', 'fundamental', 'phrase', 'assumption',
                    'sand', 'designer', 'planning', 'leading', 'mode', 'track', 'respect', 'widely', 'occasion',
                    'pose', 'approximately', 'retire', 'elsewhere', 'festival', 'cap', 'secure', 'attach', 'mechanism',
                    'intention', 'scenario', 'yell', 'incredible', 'spanish', 'strongly', 'racial', 'transportation',
                    'pot', 'boyfriend', 'consideration', 'prior', 'retirement', 'rarely', 'joint', 'doubt', 'preserve',
                    'enormous', 'cigarette', 'factory', 'valuable', 'clip', 'electric', 'giant', 'slave', 'submit',
                    'effectively', 'christian', 'monitor', 'wonder', 'resolve', 'remaining', 'participation', 'stream',
                    'rid', 'origin', 'teen', 'particular', 'congressional', 'bind', 'coat', 'tower', 'license',
                    'twitter', 'impose', 'innocent', 'curriculum', 'mail', 'estimate', 'insight', 'investigator',
                    'virus', 'hurricane', 'accurate', 'provision', 'strike', 'communicate', 'cross', 'vary', 'jacket',
                    'increasing', 'green', 'equally', 'pay', 'in', 'light', 'implication', 'fiction', 'protest',
                    'mama', 'imply', 'twin', 'pant', 'another', 'ahead', 'bend', 'shock', 'exercise', 'criteria',
                    'arab', 'dirty', 'ring', 'toy', 'potentially', 'assault', 'peak', 'anger', 'boot', 'dramatic',
                    'peer', 'enhance', 'math', 'slide', 'favor', 'pink', 'dust', 'aunt', 'lost', 'prospect', 'mood',
                    'mm-hmm', 'settlement', 'rather', 'justify', 'depth', 'juice', 'formal', 'virtually', 'gallery',
                    'tension', 'throat', 'draft', 'reputation', 'index', 'normally', 'mess', 'joy', 'steel', 'motor',
                    'enterprise', 'salary', 'moreover', 'giant', 'cousin', 'ordinary', 'graduate', 'dozen',
                    'evolution', 'so-called', 'helpful', 'competitive', 'lovely', 'fishing', 'anxiety', 'professional',
                    'carbon', 'essay', 'islamic', 'honor', 'drama', 'odd', 'evil', 'stranger', 'belt', 'urge', 'toss',
                    'fifth', 'formula', 'potato', 'monster', 'smoke', 'telephone', 'rape', 'palm', 'jet', 'navy',
                    'excited', 'plot', 'angle', 'criticize', 'prisoner', 'discipline', 'negotiation', 'damn', 'butter',
                    'desert', 'complicated', 'prize', 'blind', 'assign', 'bullet', 'awareness', 'sequence',
                    'illustrate', 'drop', 'pack', 'provider', 'fucking', 'minor', 'activist', 'poem', 'vacation',
                    'weigh', 'gang', 'privacy', 'clock', 'arrange', 'penalty', 'stomach', 'concert', 'originally',
                    'statistics', 'electronic', 'properly', 'bureau', 'wolf', 'and/or', 'classic', 'recommendation',
                    'exciting', 'maker', 'dear', 'impression', 'broken', 'battery', 'narrative', 'process', 'arise',
                    'kid', 'sake', 'delivery', 'forgive', 'visible', 'heavily', 'junior', 'rep', 'diversity', 'string',
                    'lawsuit', 'latter', 'cute', 'deputy', 'restore', 'buddy', 'psychological', 'besides', 'intense',
                    'friendly', 'evil', 'lane', 'hungry', 'bean', 'sauce', 'print', 'dominate', 'testing', 'trick',
                    'fantasy', 'absence', 'offensive', 'symbol', 'recognition', 'detect', 'tablespoon', 'construct',
                    'hmm', 'arrest', 'approval', 'aids', 'whereas', 'defensive', 'independence', 'apologize', 'top',
                    'asian', 'rose', 'ghost', 'involvement', 'permanent', 'wire', 'whisper', 'mouse', 'airline',
                    'founder', 'objective', 'nowhere', 'alternative', 'phenomenon', 'evolve', 'not', 'exact', 'silver',
                    'cent', 'universal', 'teenager', 'crucial', 'viewer', 'schedule', 'ridiculous', 'chocolate',
                    'sensitive', 'bottom', 'grandmother', 'missile', 'roughly', 'constitutional', 'adventure',
                    'genetic', 'advance', 'related', 'swing', 'ultimate', 'manufacturer', 'unknown', 'wipe', 'crop',
                    'survival', 'line', 'dimension', 'resist', 'request', 'roll', 'shape', 'darkness', 'guarantee',
                    'historic', 'educator', 'rough', 'personnel', 'race', 'confront', 'terrorist', 'royal', 'elite',
                    'occupy', 'emphasis', 'wet', 'destruction', 'raw', 'inner', 'proceed', 'violate', 'chart', 'pace',
                    'finance', 'champion', 'snap', 'suspect', 'advise', 'initially', 'advanced', 'unlikely', 'barrier',
                    'advocate', 'label', 'access', 'horrible', 'burden', 'violation', 'unlike', 'idiot', 'lifetime',
                    'working', 'fund', 'ongoing', 'react', 'routine', 'presentation', 'supply', 'gear', 'photograph',
                    'mexican', 'stadium', 'translate', 'mortgage', 'sheriff', 'clinic', 'spin', 'coalition',
                    'naturally', 'hopefully', 'mix', 'menu', 'smooth', 'advertising', 'interpret', 'plant', 'dismiss',
                    'muslim', 'apparent', 'arrangement', 'incorporate', 'split', 'brilliant', 'storage', 'framework',
                    'honestly', 'chase', 'sigh', 'assure', 'utility', 'taste', 'aggressive', 'cookie', 'terror',
                    'free', 'worth', 'wealthy', 'update', 'forum', 'alliance', 'possess', 'empire', 'curious', 'corn',
                    'neither', 'calculate', 'hurry', 'testimony', 'elementary', 'transfer', 'stake', 'precisely',
                    'bite', 'given', 'substantial', 'depending', 'glance', 'tissue', 'concentration', 'developer',
                    'found', 'ballot', 'consume', 'overcome', 'biological', 'chamber', 'similarly', 'stick', 'dare',
                    'developing', 'tiger', 'ratio', 'lover', 'expansion', 'encounter', 'occasionally', 'unemployment',
                    'pet', 'awful', 'laboratory', 'administrator', 'wind', 'quarterback', 'rocket', 'preparation',
                    'relative', 'confident', 'strategic', 'marine', 'quote', 'publisher', 'innovation', 'highlight',
                    'nut', 'fighter', 'rank', 'electricity', 'instance', 'fortune', 'freeze', 'variation', 'armed',
                    'negotiate', 'laughter', 'wisdom', 'correspondent', 'mixture', 'murder', 'assistant', 'retain',
                    'tomato', 'indian', 'testify', 'ingredient', 'since', 'galaxy', 'qualify', 'scheme', 'gop',
                    'shame', 'concentrate', 'contest', 'introduction', 'boundary', 'tube', 'versus', 'chef',
                    'regularly', 'ugly', 'screw', 'load', 'tongue', 'palestinian', 'fiscal', 'creek', 'hip',
                    'accompany', 'decline', 'terrorism', 'respondent', 'narrator', 'voting', 'refugee', 'assembly',
                    'fraud', 'limitation', 'house', 'partnership', 'store', 'crash', 'surprise', 'representation',
                    'hold', 'ministry', 'flat', 'wise', 'witness', 'excuse', 'register', 'comedy', 'purchase', 'tap',
                    'infrastructure', 'organic', 'islam', 'diverse', 'favor', 'intellectual', 'tight', 'port', 'fate',
                    'market', 'absolute', 'dialogue', 'plus', 'frequency', 'tribe', 'external', 'appointment',
                    'convert', 'surprising', 'mobile', 'establishment', 'worried', 'bye', 'shopping', 'celebrity',
                    'congressman', 'impress', 'taxpayer', 'adapt', 'publicly', 'pride', 'clothing', 'rapidly',
                    'domain', 'mainly', 'ceiling', 'alter', 'shelter', 'random', 'obligation', 'shower', 'beg',
                    'asleep', 'musician', 'extraordinary', 'dirt', 'pc', 'bell', 'smell', 'damage', 'ceremony', 'clue',
                    'guideline', 'comfort', 'near', 'pregnancy', 'borrow', 'conventional', 'tourist', 'incentive',
                    'custom', 'cheek', 'tournament', 'double', 'satellite', 'nearby', 'comprehensive', 'stable',
                    'medication', 'script', 'educate', 'efficient', 'risk', 'welcome', 'scare', 'psychology', 'logic',
                    'economics', 'update', 'nevertheless', 'devil', 'thirty', 'beat', 'charity', 'fiber', 'wave',
                    'ideal', 'friendship', 'net', 'motivation', 'differently', 'reserve', 'observer', 'humanity',
                    'survivor', 'fence', 'quietly', 'humor', 'major', 'funeral', 'spokesman', 'extension', 'loose',
                    'sink', 'historian', 'ruin', 'balance', 'chemical', 'singer', 'drunk', 'swim', 'onion',
                    'specialist', 'missing', 'white', 'pan', 'distribute', 'silly', 'deck', 'reflection', 'shortly',
                    'database', 'flow', 'remote', 'permission', 'remarkable', 'everyday', 'lifestyle', 'sweep',
                    'naked', 'sufficient', 'lion', 'consumption', 'capability', 'practice', 'emission', 'sidebar',
                    'crap', 'dealer', 'measurement', 'vital', 'impressive', 'bake', 'fantastic', 'adviser', 'yield',
                    'mere', 'imagination', 'radical', 'tragedy', 'scary', 'consultant', 'correct', 'lieutenant',
                    'upset', 'attractive', 'acre', 'drawing', 'defeat', 'newly', 'scandal', 'ambassador', 'ooh',
                    'spot', 'content', 'round', 'bench', 'guide', 'counter', 'chemical', 'odds', 'rat', 'horror',
                    'appeal', 'vulnerable', 'prevention', 'square', 'segment', 'ban', 'tail', 'constitute', 'badly',
                    'bless', 'literary', 'magic', 'implementation', 'legitimate', 'slight', 'crash', 'strip',
                    'desperate', 'distant', 'preference', 'politically', 'feedback', 'health-care', 'criminal', 'can',
                    'italian', 'detailed', 'buyer', 'wrong', 'cooperation', 'profession', 'incredibly', 'orange',
                    'killing', 'sue', 'photographer', 'running', 'engagement', 'differ', 'paint', 'pitch', 'extensive',
                    'salad', 'stair', 'notice', 'grace', 'divorce', 'vessel', 'pig', 'assignment', 'distinction',
                    'fit', 'circuit', 'acid', 'canadian', 'flee', 'efficiency', 'memorial', 'proposed', 'blue',
                    'entity', 'iphone', 'punishment', 'pause', 'pill', 'rub', 'romantic', 'myth', 'economist', 'latin',
                    'decent', 'assistant', 'craft', 'poetry', 'terrorist', 'thread', 'wooden', 'confuse', 'subject',
                    'privilege', 'coal', 'fool', 'cow', 'characterize', 'pie', 'decrease', 'resort', 'legacy', 're',
                    'stress', 'frankly', 'matter', 'cancel', 'derive', 'dumb', 'scope', 'formation', 'grandfather',
                    'hence', 'wish', 'margin', 'wound', 'exhibition', 'legislature', 'furthermore', 'portrait',
                    'catholic', 'sustain', 'uniform', 'painful', 'loud', 'miracle', 'harm', 'zero', 'tactic', 'mask',
                    'calm', 'inflation', 'hunting', 'physically', 'final', 'flesh', 'temporary', 'fellow', 'nerve',
                    'lung', 'steady', 'headline', 'sudden', 'successfully', 'defendant', 'pole', 'satisfy', 'entrance',
                    'aircraft', 'withdraw', 'cabinet', 'relative', 'repeatedly', 'happiness', 'admission',
                    'correlation', 'proportion', 'dispute', 'candy', 'reward', 'counselor', 'recording', 'pile',
                    'explosion', 'appoint', 'couch', 'cognitive', 'furniture', 'significance', 'grateful', 'magic',
                    'suit', 'commissioner', 'shelf', 'tremendous', 'warrior', 'physics', 'garage', 'flavor', 'squeeze',
                    'prominent', 'fifty', 'fade', 'oven', 'satisfaction', 'discrimination', 'recession', 'allegation',
                    'boom', 'weekly', 'lately', 'restriction', 'diamond', 'document', 'crack', 'conviction', 'heel',
                    'fake', 'fame', 'shine', 'swing', 'playoff', 'actress', 'cheat', 'format', 'controversy', 'auto',
                    'grant', 'grocery', 'headquarters', 'rip', 'rank', 'shade', 'regulate', 'meter', 'olympic', 'pipe',
                    'patient', 'celebration', 'handful', 'copyright', 'dependent', 'signature', 'bishop', 'strengthen',
                    'soup', 'entitle', 'whoever', 'carrier', 'anniversary', 'pizza', 'ethics', 'legend', 'eagle',
                    'scholarship', 'crack', 'research', 'membership', 'standing', 'possession', 'treaty', 'partly',
                    'consciousness', 'manufacturing', 'announcement', 'tire', 'no', 'makeup', 'pop', 'prediction',
                    'stability', 'trace', 'norm', 'irish', 'genius', 'gently', 'operator', 'mall', 'rumor', 'poet',
                    'tendency', 'subsequent', 'alien', 'explode', 'cool', 'controversial', 'maintenance', 'courage',
                    'exceed', 'tight', 'principal', 'vaccine', 'identification', 'sandwich', 'bull', 'lens', 'twelve',
                    'mainstream', 'presidency', 'integrity', 'distinct', 'intelligent', 'secondary', 'bias',
                    'hypothesis', 'fifteen', 'nomination', 'delay', 'adjustment', 'sanction', 'render', 'shop',
                    'acceptable', 'mutual', 'high', 'examination', 'meaningful', 'communist', 'superior', 'currency',
                    'collective', 'tip', 'flame', 'guitar', 'doctrine', 'palestinian', 'float', 'commerce', 'invent',
                    'robot', 'rapid', 'plain', 'respectively', 'particle', 'across', 'glove', 'till', 'edit',
                    'moderate', 'jazz', 'infant', 'summary', 'server', 'leather', 'radiation', 'prompt', 'function',
                    'composition', 'operating', 'assert', 'case', 'discourse', 'loud', 'dump', 'net', 'wildlife',
                    'soccer', 'complex', 'mandate', 'monitor', 'downtown', 'nightmare', 'barrel', 'homeless', 'globe',
                    'uncomfortable', 'execute', 'feel', 'trap', 'gesture', 'pale', 'tent', 'receiver', 'horizon',
                    'diagnosis', 'considerable', 'gospel', 'automatically', 'fighting', 'stroke', 'wander', 'duck',
                    'grain', 'beast', 'concern', 'remark', 'fabric', 'civilization', 'warm', 'corruption', 'collapse',
                    "ma'am", 'greatly', 'workshop', 'inquiry', 'cd', 'admire', 'exclude', 'rifle', 'closet',
                    'reporting', 'curve', 'patch', 'touchdown', 'experimental', 'earnings', 'hunter', 'fly', 'tunnel',
                    'corps', 'behave', 'rent', 'german', 'motivate', 'attribute', 'elderly', 'virtual', 'minimum',
                    'weakness', 'progressive', 'doc', 'medium', 'virtue', 'ounce', 'collapse', 'delay', 'athletic',
                    'confusion', 'legislative', 'facilitate', 'midnight', 'deer', 'way', 'undergo', 'heritage',
                    'summit', 'sword', 'telescope', 'donate', 'blade', 'toe', 'agriculture', 'park', 'enforce',
                    'recruit', 'favor', 'dose', 'concerning', 'integrate', 'rate', 'pitch', 'prescription', 'retail',
                    'adoption', 'monthly', 'deadly', 'grave', 'rope', 'reliable', 'label', 'transaction', 'lawn',
                    'consistently', 'mount', 'bubble', 'briefly', 'absorb', 'princess', 'log', 'blanket', 'laugh',
                    'kingdom', 'anticipate', 'bug', 'primary', 'dedicate', 'nominee', 'transformation', 'temple',
                    'sense', 'arrival', 'frustration', 'changing', 'demonstration', 'pollution', 'poster', 'nail',
                    'nonprofit', 'cry', 'guidance', 'exhibit', 'pen', 'interrupt', 'lemon', 'bankruptcy', 'resign',
                    'dominant', 'invasion', 'sacred', 'replacement', 'portray', 'hunt', 'distinguish', 'melt',
                    'consensus', 'kiss', 'french', 'hardware', 'rail', 'cold', 'mate', 'dry', 'korean', 'cabin',
                    'dining', 'liberal', 'snake', 'tobacco', 'orientation', 'trigger', 'wherever', 'seize', 'abuse',
                    'mess', 'punish', 'sexy', 'depict', 'input', 'seemingly', 'widespread', 'competitor', 'flip',
                    'freshman', 'donation', 'administrative', 'donor', 'gradually', 'overlook', 'toilet', 'pleased',
                    'resemble', 'ideology', 'glory', 'maximum', 'organ', 'skip', 'starting', 'brush', 'brick', 'gut',
                    'reservation', 'rebel', 'disappointed', 'oak', 'valid', 'instructor', 'rescue', 'racism',
                    'pension', 'diabetes', 'overall', 'cluster', 'eager', 'marijuana', 'combat', 'praise', 'costume',
                    'sixth', 'frequent', 'inspiration', 'orange', 'concrete', 'cooking', 'conspiracy', 'trait', 'van',
                    'institutional', 'garlic', 'drinking', 'response', 'crystal', 'stretch', 'pro', 'associate',
                    'helicopter', 'counsel', 'equation', 'roman', 'sophisticated', 'timing', 'pope', 'opera',
                    'ethical', 'mount', 'indication', 'motive', 'porch', 'reinforce', 'gaze', 'ours', 'lap', 'written',
                    'reverse', 'starter', 'injure', 'chronic', 'continued', 'exclusive', 'colonel', 'copy', 'beef',
                    'abroad', 'thanksgiving', 'intensity', 'desire', 'cave', 'basement', 'associated', 'unlike',
                    'fascinating', 'interact', 'illustration', 'daily', 'essence', 'container', 'driving', 'stuff',
                    'dynamic', 'gym', 'bat', 'plead', 'promotion', 'uncertainty', 'ownership', 'officially', 'tag',
                    'documentary', 'stem', 'flood', 'guilt', 'inside', 'alarm', 'turkey', 'conduct', 'diagnose',
                    'precious', 'swallow', 'initiate', 'fitness', 'restrict', 'gulf', 'advocate', 'mommy',
                    'unexpected', 'shrug', 'agricultural', 'sacrifice', 'spectrum', 'dragon', 'bacteria', 'shore',
                    'pastor', 'cliff', 'ship', 'adequate', 'rape', 'addition', 'tackle', 'occupation', 'compose',
                    'slice', 'brave', 'military', 'stimulus', 'patent', 'powder', 'harsh', 'chaos', 'kit', 'this',
                    'piano', 'surprisingly', 'lend', 'correctly', 'project', 'govern', 'modest', 'shared',
                    'psychologist', 'servant', 'overwhelming', 'elevator', 'hispanic', 'divine', 'transmission',
                    'butt', 'commonly', 'cowboy', 'ease', 'intent', 'counseling', 'gentle', 'rhythm', 'short',
                    'complexity', 'nonetheless', 'effectiveness', 'lonely', 'statistical', 'longtime', 'strain',
                    'firm', 'garbage', 'devote', 'speed', 'venture', 'lock', 'aide', 'subtle', 'rod', 'top',
                    'civilian', 't-shirt', 'endure', 'civilian', 'basket', 'strict', 'loser', 'franchise', 'saint',
                    'aim', 'prosecution', 'bite', 'lyrics', 'compound', 'architecture', 'reach', 'destination', 'cope',
                    'province', 'sum', 'lecture', 'spill', 'genuine', 'upstairs', 'protest', 'trading', 'please',
                    'acceptance', 'revelation', 'march', 'indicator', 'collaboration', 'rhetoric', 'tune', 'slam',
                    'inevitable', 'monkey', 'till', 'protocol', 'productive', 'principal', 'finish', 'jeans',
                    'companion', 'convict', 'boost', 'recipient', 'practically', 'array', 'persuade', 'undermine',
                    'yep', 'ranch', 'scout', 'medal', 'endless', 'translation', 'ski', 'conservation', 'habitat',
                    'contractor', 'trailer', 'pitcher', 'towel', 'goodbye', 'harm', 'bonus', 'dramatically', 'genre',
                    'caller', 'exit', 'hook', 'behavioral', 'omit', 'pit', 'volunteer', 'boring', 'hook', 'suspend',
                    'cholesterol', 'closed', 'advertisement', 'bombing', 'consult', 'encounter', 'expertise',
                    'creator', 'peaceful', 'upset', 'provided', 'tablet', 'blow', 'ruling', 'launch', 'warming',
                    'equity', 'rational', 'classic', 'utilize', 'pine', 'past', 'bitter', 'guard', 'surgeon',
                    'affordable', 'tennis', 'artistic', 'download', 'suffering', 'accuracy', 'literacy', 'treasury',
                    'talented', 'crown', 'importantly', 'bare', 'invisible', 'sergeant', 'regulatory', 'thumb',
                    'colony', 'walking', 'accessible', 'damn', 'integration', 'spouse', 'award', 'excitement',
                    'residence', 'bold', 'adolescent', 'greek', 'doll', 'oxygen', 'finance', 'gravity', 'functional',
                    'palace', 'echo', 'cotton', 'rescue', 'estimated', 'program', 'endorse', 'lawmaker',
                    'determination', 'flash', 'simultaneously', 'dynamics', 'shell', 'hint', 'frame', 'administer',
                    'rush', 'christianity', 'distract', 'ban', 'alleged', 'statute', 'value', 'biology', 'republican',
                    'follower', 'nasty', 'evident', 'prior', 'confess', 'eligible', 'picture', 'rock', 'trap',
                    'consent', 'pump', 'down', 'bloody', 'hate', 'occasional', 'trunk', 'prohibit', 'sustainable',
                    'belly', 'banking', 'asshole', 'journalism', 'flash', 'average', 'obstacle', 'ridge', 'heal',
                    'bastard', 'cheer', 'apology', 'tumor', 'architect', 'wrist', 'harbor', 'handsome', 'bullshit',
                    'realm', 'bet', 'twist', 'inspector', 'surveillance', 'trauma', 'rebuild', 'romance', 'gross',
                    'deadline', 'age', 'classical', 'convey', 'compensation', 'insect', 'debate', 'output',
                    'parliament', 'suite', 'opposed', 'fold', 'separation', 'demon', 'eating', 'structural', 'besides',
                    'equality', 'logical', 'probability', 'await', 'generous', 'acquisition', 'custody', 'compromise',
                    'greet', 'trash', 'judicial', 'earthquake', 'insane', 'realistic', 'wake', 'assemble', 'necessity',
                    'horn', 'parameter', 'grip', 'modify', 'signal', 'sponsor', 'mathematics', 'hallway',
                    'african-american', 'any', 'liability', 'crawl', 'theoretical', 'condemn', 'fluid', 'homeland',
                    'technological', 'exam', 'anchor', 'spell', 'considering', 'conscious', 'vitamin', 'known',
                    'hostage', 'reserve', 'actively', 'mill', 'teenage', 'respect', 'retrieve', 'processing',
                    'sentiment', 'offering', 'oral', 'convinced', 'photography', 'coin', 'laptop', 'bounce',
                    'goodness', 'affiliation', 'punch', 'burst', 'bee', 'blessing', 'command', 'continuous', 'above',
                    'landing', 'repair', 'worry', 'ritual', 'bath', 'sneak', 'historically', 'mud', 'scan', 'reminder',
                    'hers', 'slavery', 'supervisor', 'quantity', 'olympics', 'pleasant', 'slope', 'skirt', 'outlet',
                    'curtain', 'declaration', 'seal', 'immune', 'switch', 'calendar', 'paragraph', 'identical',
                    'credit', 'regret', 'quest', 'flat', 'entrepreneur', 'specify', 'stumble', 'clay', 'noon', 'last',
                    'strip', 'elbow', 'outstanding', 'uh-huh', 'unity', 'rent', 'manipulate', 'airplane', 'portfolio',
                    'mysterious', 'delicious', 'northwest', 'sweat', 'profound', 'sacrifice', 'treasure', 'flour',
                    'lightly', 'rally', 'default', 'alongside', 'plain', 'hug', 'isolate', 'exploration', 'secure',
                    'limb', 'enroll', 'outer', 'charter', 'southwest', 'escape', 'arena', 'witch', 'upcoming', 'forty',
                    'someday', 'unite', 'courtesy', 'statue', 'fist', 'castle', 'precise', 'squad', 'cruise', 'joke',
                    'legally', 'embassy', 'patience', 'medium', 'thereby', 'bush', 'purple', 'peer', 'electrical',
                    'outfit', 'cage', 'retired', 'shark', 'lobby', 'sidewalk', 'near', 'runner', 'ankle', 'attraction',
                    'fool', 'artificial', 'mercy', 'indigenous', 'slap', 'tune', 'dancer', 'candle', 'sexually',
                    'needle', 'hidden', 'chronicle', 'suburb', 'toxic', 'underlying', 'sensor', 'deploy', 'debut',
                    'star', 'magnitude', 'suspicion', 'pro', 'colonial', 'icon', 'grandma', 'info', 'jurisdiction',
                    'iranian', 'senior', 'parade', 'seal', 'archive', 'gifted', 'rage', 'outdoor', 'ending', 'loop',
                    'altogether', 'chase', 'burning', 'reception', 'local', 'crush', 'premise', 'deem', 'automatic',
                    'whale', 'mechanical', 'credibility', 'drain', 'drift', 'loyalty', 'promising', 'tide', 'traveler',
                    'grief', 'metaphor', 'skull', 'pursuit', 'therapist', 'backup', 'workplace', 'instinct', 'export',
                    'bleed', 'shock', 'seventh', 'fixed', 'broadcast', 'disclose', 'execution', 'pal', 'chuckle',
                    'pump', 'density', 'correction', 'representative', 'jump', 'repair', 'kinda', 'relieve',
                    'teammate', 'brush', 'corridor', 'russian', 'enthusiasm', 'extended', 'root', 'alright', 'panic',
                    'pad', 'bid', 'mild', 'productivity', 'guess', 'tuck', 'defeat', 'railroad', 'frozen', 'minimize',
                    'amid', 'inspection', 'cab', 'expected', 'nonsense', 'leap', 'draft', 'rider', 'theology',
                    'terrific', 'accent', 'invitation', 'reply', 'israeli', 'liar', 'oversee', 'awkward',
                    'registration', 'suburban', 'handle', 'momentum', 'instantly', 'clerk', 'chin', 'hockey', 'laser',
                    'proposition', 'rob', 'beam', 'ancestor', 'creativity', 'verse', 'casual', 'objection', 'clever',
                    'given', 'shove', 'revolutionary', 'carbohydrate', 'steam', 'reportedly', 'glance', 'forehead',
                    'resume', 'slide', 'sheep', 'good', 'carpet', 'cloth', 'interior', 'full-time', 'running',
                    'questionnaire', 'compromise', 'departure', 'behalf', 'graph', 'diplomatic', 'thief', 'herb',
                    'subsidy', 'cast', 'fossil', 'patrol', 'pulse', 'mechanic', 'cattle', 'screening', 'continuing',
                    'electoral', 'supposedly', 'dignity', 'prophet', 'commentary', 'sort', 'spread', 'serving',
                    'safely', 'homework', 'allegedly', 'android', 'alpha', 'insert', 'mortality', 'contend',
                    'elephant', 'solely', 'hurt', 'continent', 'attribute', 'ecosystem', 'leave', 'nearby', 'olive',
                    'syndrome', 'minimum', 'catch', 'abstract', 'accusation', 'coming', 'sock', 'pickup', 'shuttle',
                    'improved', 'calculation', 'innovative', 'demographic', 'accommodate', 'jaw', 'unfair', 'tragic',
                    'comprise', 'faster', 'nutrition', 'mentor', 'stance', 'rabbit', 'pause', 'dot', 'contributor',
                    'cooperate', 'disk', 'hesitate', 'regard', 'offend', 'exploit', 'compel', 'likelihood', 'sibling',
                    'southeast', 'gorgeous', 'undertake', 'painter', 'residential', 'counterpart', 'believer', 'lamp',
                    'inmate', 'thoroughly', 'trace', 'freak', 'filter', 'pillow', 'orbit', 'purse', 'likewise',
                    'cease', 'passing', 'feed', 'vanish', 'instructional', 'clause', 'mentally', 'model', 'left',
                    'pond', 'neutral', 'shield', 'popularity', 'cartoon', 'authorize', 'combined', 'exhibit', 'sink',
                    'graphic', 'darling', 'traditionally', 'vendor', 'poorly', 'conceive', 'opt', 'descend', 'firmly',
                    'beloved', 'openly', 'gathering', 'alien', 'stem', 'fever', 'preach', 'interfere', 'arrow',
                    'required', 'capitalism', 'kick', 'fork', 'survey', 'meantime', 'presumably', 'position', 'racist',
                    'stay', 'illusion', 'removal', 'anxious', 'arab', 'organism', 'awake', 'sculpture', 'spare',
                    'marine', 'harassment', 'drum', 'diminish', 'helmet', 'level', 'certificate', 'tribal', 'bad',
                    'mmm', 'sadly', 'cart', 'spy', 'sunlight', 'delete', 'rookie', 'clarify', 'hunger', 'practitioner',
                    'performer', 'protective', 'jar', 'programming', 'dawn', 'salmon', 'census', 'pick',
                    'accomplishment', 'conscience', 'fortunately', 'minimal', 'molecule', 'supportive', 'sole',
                    'threshold', 'inventory', 'comply', 'monetary', 'transport', 'shy', 'drill', 'influential',
                    'verbal', 'reward', 'ranking', 'gram', 'grasp', 'puzzle', 'envelope', 'heat', 'classify', 'enact',
                    'unfortunate', 'scatter', 'cure', 'time', 'dear', 'slice', 'readily', 'damn', 'discount',
                    'addiction', 'emerging', 'worthy', 'marker', 'juror', 'mention', 'blend', 'businessman', 'premium',
                    'retailer', 'charge', 'liver', 'pirate', 'protester', 'outlook', 'elder', 'gallon', 'additionally',
                    'ignorance', 'chemistry', 'sometime', 'weed', 'babe', 'fraction', 'cook', 'conversion', 'object',
                    'tolerate', 'trail', 'drown', 'merit', 'citizenship', 'coordinator', 'validity', 'european',
                    'lightning', 'turtle', 'ambition', 'worldwide', 'sail', 'added', 'delicate', 'comic', 'soap',
                    'hostile', 'instruct', 'shortage', 'useless', 'booth', 'diary', 'gasp', 'suspicious', 'transit',
                    'excite', 'publishing', 'curiosity', 'grid', 'rolling', 'bow', 'cruel', 'disclosure', 'rival',
                    'denial', 'secular', 'flood', 'speculation', 'sympathy', 'tender', 'inappropriate', "o'clock",
                    'sodium', 'divorce', 'spring', 'bang', 'challenging', 'ipad', 'sack', 'barn', 'reliability',
                    'hormone', 'footage', 'carve', 'alley', 'ease', 'coastal', 'cafe', 'partial', 'flexible',
                    'experienced', 'mixed', 'vampire', 'optimistic', 'dessert', 'well-being', 'northeast',
                    'specialize', 'fleet', 'availability', 'compliance', 'pin', 'pork', 'astronomer', 'like', 'forbid',
                    'installation', 'boil', 'nest', 'exclusively', 'goat', 'shallow', 'equip', 'equivalent', 'betray',
                    'willingness', 'banker', 'interval', 'gasoline', 'encouraging', 'rain', 'exchange', 'bucket',
                    'theft', 'laundry', 'constraint', 'dying', 'hatred', 'jewelry', 'migration', 'invention', 'loving',
                    'revenge', 'unprecedented', 'outline', 'sheer', 'halloween', 'sweetheart', 'spit', 'lazy',
                    'intimate', 'defender', 'technically', 'battle', 'cure', 'peanut', 'unclear', 'piss', 'workout',
                    'wilderness', 'compelling', 'eleven', 'arm', 'backyard', 'alike', 'partially', 'transport',
                    'guardian', 'passionate', 'scripture', 'midst', 'ideological', 'apart', 'thrive', 'sensitivity',
                    'trigger', 'emotionally', 'ignorant', 'explicitly', 'unfold', 'headache', 'eternal', 'chop', 'ego',
                    'spectacular', 'deposit', 'verdict', 'regard', 'accountability', 'nominate', 'civic', 'uncover',
                    'critique', 'aisle', 'tropical', 'annually', 'eighth', 'blast', 'corrupt', 'compassion', 'scratch',
                    'verify', 'offender', 'inherit', 'strive', 'downtown', 'chunk', 'appreciation', 'canvas', 'punch',
                    'short-term', 'proceedings', 'magical', 'loyal', 'aah', 'desperately', 'throne', 'brutal', 'spite',
                    'propaganda', 'irony', 'soda', 'projection', 'dutch', 'parental', 'disabled', 'collector',
                    're-election', 'disappointment', 'comic', 'aid', 'happily', 'steep', 'fancy', 'counter',
                    'listener', 'whip', 'public', 'drawer', 'heck', 'developmental', 'ideal', 'ash', 'socially',
                    'courtroom', 'stamp', 'solo', 'trainer', 'induce', 'anytime', 'morality', 'syrian', 'pipeline',
                    'bride', 'instant', 'spark', 'doorway', 'interface', 'learner', 'casino', 'placement', 'cord',
                    'fan', 'conception', 'flexibility', 'thou', 'tax', 'elegant', 'flaw', 'locker', 'peel', 'campaign',
                    'twist', 'spell', 'objective', 'plea', 'goddamn', 'import', 'stack', 'gosh', 'philosophical',
                    'junk', 'bicycle', 'vocal', 'chew', 'destiny', 'ambitious', 'unbelievable', 'vice', 'halfway',
                    'jealous', 'sphere', 'invade', 'sponsor', 'excessive', 'countless', 'sunset', 'interior',
                    'accounting', 'faithful', 'freely', 'extract', 'adaptation', 'ray', 'depressed', 'emperor',
                    'wagon', 'columnist', 'jungle', 'embarrassed', 'trillion', 'breeze', 'blame', 'foster', 'venue',
                    'discourage', 'disturbing', 'riot', 'isolation', 'explicit', 'commodity', 'attendance', 'tab',
                    'consequently', 'dough', 'novel', 'streak', 'silk', 'similarity', 'steak', 'dancing', 'petition',
                    'viable', 'breathing', 'mm', 'balloon', 'monument', 'try', 'cue', 'sleeve', 'toll', 'reluctant',
                    'warrant', 'stiff', 'tattoo', 'softly', 'sudden', 'graduation', 'japanese', 'deliberately',
                    'consecutive', 'upgrade', 'associate', 'accurately', 'strictly', 'leak', 'casualty', 'risky',
                    'banana', 'blank', 'beneficial', 'shrink', 'chat', 'rack', 'rude', 'usage', 'testament', 'browser',
                    'processor', 'thigh', 'perceived', 'yield', 'talking', 'merchant', 'quantum', 'eyebrow',
                    'surrounding', 'vocabulary', 'ashamed', 'eh', 'radar', 'stunning', 'murderer', 'burger', 'collar',
                    'align', 'textbook', 'sensation', 'afterward', 'charm', 'sunny', 'hammer', 'keyboard', 'persist',
                    'wheat', 'predator', 'bizarre']