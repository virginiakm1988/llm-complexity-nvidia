# Reasoning Stress Lab：繁中快速入門

本專案提供可重跑的 NVIDIA SDK 示範、互動 Colab 與評估流程。先讀 [英文 README](README.md) 中的論文／原始碼來源與方法差異。

1. 安裝 Python 3.11 以上，建立虛擬環境，再執行 `python -m pip install -e ".[tokenizer]"`。
2. 執行 `python -m project_demo demo --mode fixture --output-dir runs/fixture` 檢查完整流程。此模式是離線測試值，不是模型成績。
3. 將 API key 放在環境變數或 Colab Secrets 的 `NVIDIA_API_KEY`，勿寫入程式或 notebook。
4. 執行 `python -m project_demo demo --config configs/live-smoke.json --mode live --output-dir runs/live` 取得真實 NIM 回答。
5. 執行 `python -m project_demo evaluate --output-dir runs/live` 重算評分；查看逐筆輸出、錯誤及設定。

每次實驗使用新的輸出資料夾，避免覆寫舊證據。API 失敗會保留錯誤，不會換成離線結果。教學提供文字輸入、設定選單及結果比較；完整測試記錄在 [validation.md](docs/validation.md)。

要換資料或模型，先確認資料授權、官方 test split、tokenizer 對應及評分規則，再修改設定檔。Data Designer 的候選文字需人工確認語意後才納入實驗。
