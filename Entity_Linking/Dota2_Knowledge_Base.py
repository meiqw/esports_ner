
import json
from difflib import get_close_matches

class Dota2_Knowledge_Base():

    def __init__(self, player_file_name:str = None, team_file_name:str = None, tournament_file_name:str = None, heroes_file_name:str = None) -> None:
        if player_file_name is not None:
            players = self.load_json_file(player_file_name)
            self.players = self.expand_players(players)
        if team_file_name is not None:
            teams = self.load_json_file(team_file_name)
            self.teams = self.expand_teams(teams)
        if tournament_file_name is not None:
            tournaments = self.load_json_file(tournament_file_name)
            self.tournaments = self.expand_tournaments(tournaments)
        self.url_prefix = "https://liquipedia.net/dota2/"

    def load_json_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as json_file:
            return json.load(json_file)

    def expand_teams(self, teams):
        teams_expanded = {}
        for team in teams:
            possible_names = [team['name']]
            team['link'] = 'https://liquipedia.net/dota2/' + team['name'].replace(' ', '_')
            for alias in team['aliases']:
                possible_names.append(alias)
            words = team['name'].strip().split(' ')
            if len(words) >= 2:
                initials = ''.join([word[0].upper() for word in words])
                possible_names.append(initials)
            if team['name'].startswith('Team'):
                possible_names.append(team['name'].replace('Team', '', 1).strip())
            for possible_name in possible_names:
                teams_expanded[possible_name] = team
        return teams_expanded

    def expand_tournaments(self, tournaments):
        tournaments_expanded = {}
        for tournament in tournaments:
            tournaments_expanded[tournament['name']] = tournament
        return tournaments_expanded

    def expand_players(self, players):
        players_expanded = {}
        for player in players:
            possible_names = self.get_possible_player_aliases(player)
            for possible_name in possible_names:
                players_expanded[possible_name] = player
        return players_expanded

    def get_possible_player_aliases(self, player):
        res = []
        res.append(player['ID'])
        res.append(player['Name'])
        return res

    def get_best_key(self, query_string, keys):
        if query_string in keys:
            return query_string
        for key in keys:
            if query_string in key:
                return key
        match_res = get_close_matches(query_string, keys, 1)
        if len(match_res) > 0:
            return match_res[0]
        return None

    def get_matching_player(self, player_name):
        best_guess = self.get_best_key(player_name, self.players.keys())
        if best_guess is not None:
            return self.players[best_guess]
        return None

    def get_matching_team(self, team_name):
        best_guess = self.get_best_key(team_name, self.teams.keys())
        if best_guess is not None:
            return self.teams[best_guess]
        return None

    def get_matching_tournament(self, tournament_name):
        best_guess = self.get_best_key(tournament_name, self.tournaments.keys())
        if best_guess is not None:
            return self.tournaments[best_guess]
        return None
