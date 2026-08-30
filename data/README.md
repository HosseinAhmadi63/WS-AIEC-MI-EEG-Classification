# Public EEG data

The repository does not redistribute EEG recordings. scripts/cache_data.py
uses MOABB 1.5.0 to download the official files and create ordered participant
caches below `data/cache/<run-key>/`.

| Dataset | Source | Executable cohort |
|---|---|---|
| BNCI2014-001 | BNCI Horizon 001-2014 | 9 participants, 2 sessions, 144 trials per class |
| BNCI2014-002 | BNCI Horizon 002-2014 | 14 participants, 1 session, 80 trials per class |
| BNCI2014-004 | BNCI Horizon 004-2014 | 9 participants, 5 sessions, 360 trials per class |
| BNCI2015-001 | BNCI Horizon 001-2015 | 12 participants, first 2 sessions, 200 trials per class |
| Zhou2016 | Figshare file 3662952 | 4 participants, 3 sessions, 150 available trials per class |
| AlexMI | Zenodo record 806023 | 8 participants, 1 session, 20 trials per class |

BNCI catalogue: <https://bnci-horizon-2020.eu/database/data-sets>

Expected cache shapes per participant:

~~~text
BNCI2014_001  576 x 22 x 1000
BNCI2014_002  160 x 15 x 2560
BNCI2014_004  720 x  3 x 1125
BNCI2015_001  400 x 13 x 2560
Zhou2016       450 x 14 x 1250
AlexMI          60 x 16 x 1536
~~~

The loader rejects a cache whose dataset identity, participant identity, class
order, EEG channel count, sampling rate, epoch length, or trial counts differ
from configs/paper.yaml.
