"""Read-only public dashboard contracts. All demo fixtures are explicitly synthetic."""
from fastapi.testclient import TestClient

from news_predictor.web.app import app
from news_predictor.web.data import DEMO_AS_OF, assets, get_asset

client = TestClient(app)


def test_health_and_frontend():
    r = client.get('/api/health')
    assert r.status_code == 200 and r.json()['status'] == 'ok'
    page = client.get('/')
    assert page.status_code == 200
    assert 'SYNTHETIC DEMONSTRATION' in page.text
    assert '/static/app.js' in page.text
    assert client.get('/static/styles.css').status_code == 200
    assert client.get('/docs').status_code == 200


def test_overview_is_labeled_synthetic_and_has_all_assets():
    r = client.get('/api/overview?mode=demo&sort_by=mentions')
    assert r.status_code == 200
    data = r.json()
    assert data['is_demo'] is True
    assert 'SYNTHETIC' in data['notice']
    assert data['assets_available'] == len(assets()) == 5
    assert data['as_of'] == DEMO_AS_OF
    assert all(x['data_status'] == 'synthetic_demo' for x in data['assets'])
    assert all(x['predicted_next_close'] is not None for x in data['assets'])
    mentions = [x['social_mentions'] for x in data['assets']]
    assert mentions == sorted(mentions, reverse=True)
    assert 'history' not in data['assets'][0]  # summary size bounded
    assert 'evaluation' not in data['assets'][0]


def test_demo_detail_is_reproducible_and_explicitly_not_trained():
    first = client.get('/api/assets/0700.HK/analysis?mode=demo')
    second = client.get('/api/assets/0700.HK/analysis?mode=demo')
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    data = first.json()
    assert data['is_demo'] is True
    assert data['evaluation']['kind'] == 'synthetic_illustration'
    assert len(data['history']) == 70
    assert len(data['evaluation']['test_series']) == data['evaluation']['sample_count']
    assert 'no trained model' in data['prediction_status'].lower()
    assert data['source_mentions']['lihkg'] + data['source_mentions']['futu'] == data['social_mentions']


def test_contract_rejects_invalid_tickers_and_choices():
    assert client.get('/api/assets/NOT-A-STOCK/analysis?mode=demo').status_code == 404
    assert client.get('/api/assets/../../.env/analysis?mode=demo').status_code in (404, 422)
    assert client.get('/api/overview?mode=unknown').status_code == 422
    assert client.get('/api/overview?sort_by=profit').status_code == 422
    assert client.get('/api/assets/0700.HK/analysis?mode=invalid').status_code == 422


def test_real_mode_never_fabricates_missing_files(tmp_path, monkeypatch):
    import news_predictor.web.data as data
    original_root = data.repo_root
    fake_repo = tmp_path
    (fake_repo/'config').mkdir()
    (fake_repo/'config'/'assets.json').write_text((original_root()/'config'/'assets.json').read_text())
    monkeypatch.setattr(data, 'repo_root', lambda: fake_repo)
    r = client.get('/api/overview?mode=real')
    assert r.status_code == 200
    assert r.json()['is_demo'] is False
    assert r.json()['assets_available'] == 0
    assert r.json()['assets'] == []
    assert client.get('/api/assets/0700.HK/analysis?mode=real').status_code == 404
    assert get_asset('0700.HK', 'real') is None


def test_sources_contain_no_secret_material():
    r = client.get('/api/sources')
    assert r.status_code == 200
    assert r.json()['real_collection_enabled_over_http'] is False
    assert len(r.json()['sources']) == 4
    assert 'APIFY_TOKEN' not in r.text and 'EXA_API_KEY' not in r.text
