"""
研究室URL収集ストラテジ（collector）のレジストリ。

サイト構造ごとに 1 モジュールを作り、ここに登録する。各 collector モジュールは
以下を公開する:
  - NAME: str
  - matches(index_url: str) -> bool        # この collector が担当するURLか
  - async collect(browser, index_url, faculty, department) -> list[dict]

新しい研究科を追加するとき:
  - 既存のサイト構造で済む → 既存 collector の matches() を広げる
  - 構造が違う           → collectors/<name>.py を新規作成し _REGISTRY に追加する

注意: サイト構造は「研究科」より粗い単位で共通化できることが多い
（同じCMSを使う複数研究科を 1 collector でカバーできる）。
"""

from . import engineering, informatics

# 振り分け順（先に matches した collector を使う）。
# informatics を末尾に置き、既定フォールバックも兼ねる。
_REGISTRY = [engineering, informatics]

# 既定フォールバック（どの collector も matches しなかった場合）
_DEFAULT = informatics


def select_collector(index_url: str):
    """index_url を担当する collector モジュールを返す。"""
    for mod in _REGISTRY:
        if mod.matches(index_url):
            return mod
    return _DEFAULT


def list_collectors() -> list[str]:
    return [mod.NAME for mod in _REGISTRY]
