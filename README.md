# Esports contents NER and NEL

This is a Django web application that performs named entity tagging and linking in Esports text (and specifically DOTA2, League of Legends, CS:GO and Overwatch). Our refined ontology contained 5 kinds of entities: GAME, TOURN, ORG, PLAYER, and AVATAR, defined as below:

* GAME: The esports title.
* TOURN:  An esports event or league.
* ORG: The "team" in which name players play for.
* PLAYER: Individuals who play and compete on the game as a career (in other words, “pro players”).
* AVATAR: The character that a player controls. In CS:GO, it is the item that a player uses.

The search bar accepts a URL or a text snippet and the web application taggs the above entities from the text extracted.

![demo1](https://user-images.githubusercontent.com/43922422/112531324-d0e48e00-8d7d-11eb-8156-702f9e8b3179.png)

Each entity links to another page that shows all the sentences with the entity in the corpus and its related information in Liquipedia(the esports wiki) with a URL.

A mention and its alias will have the same entry and URL in Liquipedia which suggests succesful named entity linking. 

### Prerequisites

The python packages listed in requirements.txt need to be installed for the program to run.

### Installing
Use pip to install the dependencies.

```
pip install -r requirements.txt
```

Then run

```
python -m spacy download en_core_web_sm
```

to install the spaCy models required for this assignment


## How to run

```
cd Esports_NER
python manage.py runserver
```

## Contribution

The crf NER system is contributed by Meiqi Wang and Ziyu Liu, the web application django framework is contributed by Yifan Leng.

## License

This project is licensed under the MIT License 

