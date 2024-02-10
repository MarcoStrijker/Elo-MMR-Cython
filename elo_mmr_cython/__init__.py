from datetime import datetime, timezone
from calculations import Player, construct_elo_mmr_default_gaussian, Contest, simulate_contest, ContestRatingParams, construct_elo_mmr_default_fast, construct_elo_mmr_gaussian_fast

c = Contest(1)
c.push_contestant("Peter")
c.push_contestant("Paul")

print(c.standings)
elo = construct_elo_mmr_default_fast()

players = {}

simulate_contest(players, c, elo, 1000, 1, 0)


print(players)

raise NotImplementedError

cr = ContestRatingParams()
elo_mmr = construct_elo_mmr_default_fast()
player1 = Player(1000, 0, 0)
player2 = Player(1000, 0, 0)

standings = [
    (
        player1,
        0, 0  # Range of players that got or tied for first
    ),
    (
        player2,
        1, 1  # Range of players that got or tied for second
    ),
]

# Note that the contest_time does not do anything in this example
# because EloMMR.drift_per_sec defaults to 0, so contest_time
# can be omitted from the round_update call, but it is included
# here to show how it can be used.
# Do note, though, that you should either always include
# contest_time or never include it, because if you include it
# in some competitions and not others, the ratings will be skewed
# incorrectly.
contest_time = round(datetime.now(timezone.utc).timestamp())
elo_mmr.round_update(cr, standings)

contest_time += 1000
# Assumes the outcome of the next competition is the same as the
# previous, so the standings aren't changed.
elo_mmr.round_update(cr, standings)

for player in [player1, player2]:
    print("\nrating_mu, rating_sig, perf_score, place")
    for event in player.event_history:
        print(f"{event.mu}, {event.sig}, {event.perf_score}, {event.place}")
    print(f"Final rating: {player.event_history[-1].display_rating()}")
