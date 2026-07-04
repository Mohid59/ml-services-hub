"""Fast route tests - no ML model downloads required.

These cover page rendering, input validation, and the classic-ML services
(clustering, apriori) which run instantly on CPU. Transformer-backed
happy paths live in test_ml_services.py (marked slow).
"""
import io

import pytest

ALL_PAGES = ["/", "/dbscan/", "/kmeans/", "/apriori/", "/gender/",
             "/sentiment/", "/qa/", "/textgen/", "/translation/", "/ner/"]


@pytest.mark.parametrize("path", ALL_PAGES)
def test_page_renders(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert b"flash-error" not in r.data


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


def test_404_redirects_home(client):
    r = client.get("/nope/", follow_redirects=True)
    assert r.status_code == 200


# ---------- Clustering ----------

def test_dbscan_sample(client):
    r = client.post("/dbscan/", data={"use_sample": "1", "eps": "0.6",
                                      "min_samples": "3"})
    assert r.status_code == 200
    assert b"data:image/png" in r.data
    assert b"flash-error" not in r.data


def test_dbscan_rejects_text_only_csv(client):
    csv = io.BytesIO(b"name,city\nAlice,NYC\nBob,LA\nCara,SF\n")
    r = client.post("/dbscan/", data={"eps": "0.5", "min_samples": "5",
                                      "csvfile": (csv, "t.csv")},
                    content_type="multipart/form-data")
    assert b"flash-error" in r.data


def test_dbscan_rejects_bad_eps(client):
    r = client.post("/dbscan/", data={"use_sample": "1", "eps": "-1",
                                      "min_samples": "5"})
    assert b"flash-error" in r.data


def test_dbscan_rejects_wrong_filetype(client):
    r = client.post("/dbscan/", data={"eps": "0.5", "min_samples": "5",
                                      "csvfile": (io.BytesIO(b"x"), "x.txt")},
                    content_type="multipart/form-data")
    assert b"flash-error" in r.data


def test_kmeans_sample_with_elbow(client):
    r = client.post("/kmeans/", data={"use_sample": "1", "k": "3"})
    assert r.status_code == 200
    assert r.data.count(b"data:image/png") >= 2  # scatter + elbow
    assert b"flash-error" not in r.data


@pytest.mark.parametrize("k", ["1", "999", "abc"])
def test_kmeans_rejects_bad_k(client, k):
    r = client.post("/kmeans/", data={"use_sample": "1", "k": k})
    assert b"flash-error" in r.data


# ---------- Apriori ----------

def test_apriori_sample(client):
    r = client.post("/apriori/", data={"use_sample": "1",
                                       "min_support": "0.15",
                                       "min_confidence": "0.5"})
    assert r.status_code == 200
    assert b"<table" in r.data
    assert b"flash-error" not in r.data


def test_apriori_one_hot_autodetect(client):
    oh = io.BytesIO(b"milk,bread,beer\n1,1,0\n1,0,0\n0,1,1\n1,1,0\n0,0,1\n1,1,0\n")
    r = client.post("/apriori/", data={"min_support": "0.2",
                                       "min_confidence": "0.3",
                                       "csvfile": (oh, "oh.csv")},
                    content_type="multipart/form-data")
    assert b"one-hot" in r.data
    assert b"flash-error" not in r.data


def test_apriori_no_itemsets_is_friendly(client):
    r = client.post("/apriori/", data={"use_sample": "1",
                                       "min_support": "0.99",
                                       "min_confidence": "0.5"})
    assert b"flash-info" in r.data


def test_apriori_rejects_bad_support(client):
    r = client.post("/apriori/", data={"use_sample": "1", "min_support": "9"})
    assert b"flash-error" in r.data


# ---------- Empty-input guards (no model load needed) ----------

@pytest.mark.parametrize("path,field", [
    ("/ner/", "text"), ("/textgen/", "prompt"),
    ("/translation/", "text"), ("/sentiment/", "text"),
])
def test_empty_text_rejected(client, path, field):
    r = client.post(path, data={field: ""})
    assert b"flash-error" in r.data


def test_gender_rejects_non_image(client):
    r = client.post("/gender/", data={"imagefile": (io.BytesIO(b"nope"),
                                                    "x.txt")},
                    content_type="multipart/form-data")
    assert b"flash-error" in r.data
