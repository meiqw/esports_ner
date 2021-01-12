# Esports contents NER and NEL

This is a Django web application that performs named entity tagging and linking in Esports text (and specifically DOTA2, League of Legends, CS:GO and Overwatch). Our refined ontology contained 5 kinds of entities: GAME, TOURN, ORG, PLAYER, and AVATAR, defined as below:

* GAME: The esports title.
* TOURN:  An esports event or league.
* ORG: The "team" in which name players play for.
* PLAYER: Individuals who play and compete on the game as a career (in other words, “pro players”).
* AVATAR: The character that a player controls. In CS:GO, it is the item that a player uses.

The search bar accepts a URL or text snippet and the web application taggs the above entities from the text extracted. Each entity links to another page that shows its related information in Liquipedia with a URL. A mention and its alias will have the same entry and URL in Liquipedia which suggests succesful named entity linking. 

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

## License

This project is licensed under the MIT License 

