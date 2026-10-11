import os 
import time 
import random 
import requests
from rdkit import Chem, DataStructs, RDConfig, RDLogger
from rdkit.Chem import AllChem, Descriptors
import rdkit.Chem.QED as QED
import networkx as nx
import sys 
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
RDLogger.DisableLog("rdApp.*")
import sascorer as sascorer


DRD2_URLS = [

             ]

ADMET_URLS = [

              ]


REQUEST_RETRIES = max(1, int(os.getenv("PROPERTY_SERVER_RETRIES", "5")))
REQUEST_BACKOFF = float(os.getenv("PROPERTY_SERVER_BACKOFF", "0.5"))

def _post_json_with_retry(url, payload, timeout):
    last_error = None

    for attempt in range(REQUEST_RETRIES):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()

        except (requests.RequestException, ValueError) as exc:
            last_error = exc

            if attempt + 1 == REQUEST_RETRIES:
                break

            time.sleep(REQUEST_BACKOFF * (2 ** attempt))

    raise RuntimeError(
        f"{url} failed after {REQUEST_RETRIES} attempts: {last_error}"
    )


def _post_drd2_with_failover(payload, timeout):
    last_error = None

    urls = DRD2_URLS.copy()
    random.shuffle(urls)


    for url in urls:
        try:
            result = _post_json_with_retry(url, payload, timeout)
            return result
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"all DRD2 servers failed: {last_error}")


def _post_admet_with_failover(payload, timeout):
    last_error = None

    urls = ADMET_URLS.copy()
    random.shuffle(urls)


    for url in urls:
        try:
            result = _post_json_with_retry(url, payload, timeout)
            return result
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"all ADMET servers failed: {last_error}")


def tanimoto_similarity(a, b):
    if a is None or b is None: 
        return 0.0
    amol = Chem.MolFromSmiles(a)
    bmol = Chem.MolFromSmiles(b)
    if amol is None or bmol is None:
        return 0.0

    fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
    return DataStructs.TanimotoSimilarity(fp1, fp2) 

def qed(s):
    if s is None: return 0.0
    mol = Chem.MolFromSmiles(s)
    if mol is None: return 0.0
    return QED.qed(mol)

def sas(s):
    if s is None: return 0.0
    mol = Chem.MolFromSmiles(s)
    if mol is None: return 0.0
    return sascorer.calculateScore(mol)

# Modified from https://github.com/bowenliu16/rl_graph_generation
def penalized_logp(s):
    if s is None: return -100.0
    mol = Chem.MolFromSmiles(s)
    if mol is None: return -100.0

    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA = -sascorer.calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std
    return normalized_log_p + normalized_SA + normalized_cycle

def admet(s):
    try:
        j = _post_admet_with_failover(
            {"smiles": s},
            timeout=180,
        )

        return (j["bbb"], j["ames"], j["hia"])

    except Exception:
        return 0.0, 10.0, 0.0


def drd2(s):
    try:
        return _post_drd2_with_failover(
            {"smiles": s},
            timeout=60,
        )["drd2"]

    except Exception:
        return 0.0
    

def get_properties(smiles: str, properties: list[str]) -> dict[str, float]: 
    out = {}

    if any(prop in properties for prop in ("hia", "mutagenicity", "bbbp")): 
        try: 
            bbb, ames, hia = admet(smiles) 
        except Exception: 
            bbb, ames, hia = 0.0, 10.0, 0.0 

    for prop in properties: 
        if prop == "qed": 
            try:
                out[prop] = float(qed(smiles))
            except Exception:
                out[prop] = 0.0
        elif prop == "plogp": 
            try:
                out[prop] = float(penalized_logp(smiles))
            except Exception:
                print(f"Error computing penalized_logp for {smiles}")
                out[prop] = -100.0
        elif prop == "drd2": 
            try:
                out[prop] = float(drd2(smiles))
            except Exception:
                out[prop] = 0.0
        elif prop == "hia":
            try:
                out[prop] = float(hia)
            except Exception:
                out[prop] = 0.0
        elif prop == "mutagenicity":
            try:
                out[prop] = float(ames)
            except Exception:
                out[prop] = 0.0
        elif prop == "bbbp":
            try:
                out[prop] = float(bbb)
            except Exception:
                out[prop] = 0.0
    return out 

def relative_improvement(
    prop: str,
    input_val: float,
    output_val: float,
    min_denominator: float = 0.1,
) -> float:
    if input_val is None or output_val is None:
        return 0.0

    if prop == "mutagenicity":
        delta = input_val - output_val
    else:
        delta = output_val - input_val

    denominator = max(abs(input_val), min_denominator)
    return delta / denominator

def composite_ri(
    input_props: dict[str, float],
    output_props: dict[str, float],
    properties: list[str],
) -> float:
    return sum(
        relative_improvement(p, input_props[p], output_props[p])
        for p in properties
    ) / len(properties)