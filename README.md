## LinZhaiPLUS
Use a database to compare various types of housing data and find the best value for money.

# Commit Message Guidelines

```
這裡是我們的提交訊息類型及其說明，幫助維持專案的清晰性和一致性。
```

## 提交訊息格式

## 提交類型
```
- feat: 新增或修改功能（feature）
- fix : 修補 bug（bug fix）
- docs : 文件（documentation）
- style : 格式不影響程式碼運行的變動，例如：white-space、formatting、missing semi colons
- refactor : 重構，不是新增功能，也非修補 bug 的程式碼變動
- perf : 改善效能（improves performance）
- test : 增加測試（when adding missing tests）
- chore : maintain，不影響程式碼運行，建構程序或輔助工具的變動，例如修改 config、Grunt Task 任務管理工具
- revert : 撤銷回覆先前的 commit
```

## 範例
```
fix : 將資料庫回傳的異常解決。
```

## Unit Test
**請先在終端輸入，請先確認path有沒有python跟git，requirements.txt要先cd進其路徑下**
```
pip install -r requirements.txt
```

# GitHub Issue 撰寫指南

在 GitHub 和其他版本控制系統中，「issue」是一個用來追蹤 bug、功能請求、任務或其他類型的問題的工具。本指南將幫助你有效撰寫 Issue。

## 如何撰寫 Issue

### 1. 標題 (Title)
簡潔明了地描述問題或請求的內容。例如：
- `Bug: Unable to connect to SQL database using Python`
- `Feature Request: Add user authentication`

### 2. 描述 (Description)
詳細說明問題或請求的背景、上下文和具體內容。這部分應該包含：
- **問題描述**：描述你遇到的問題或希望增加的功能。
- **重現步驟 (Reproduction Steps)**：列出能重現問題的具體步驟。
- **預期行為 (Expected Behavior)**：描述你希望發生的結果。
- **實際行為 (Actual Behavior)**：描述實際發生的情況。
- **環境 (Environment)**：提供相關的環境信息，例如操作系統、Python 版本、數據庫類型等。

### 3. 標籤 (Labels)
使用標籤來分類 Issue，例如 `bug`、`enhancement`、`question` 等。

### 4. 指派 (Assignees)
如果有特定的人負責該 Issue，可以將其指派給該用戶。

### 5. 里程碑 (Milestones)
如果有特定的版本或里程碑可以關聯，可以將其附加到該 Issue。

## 範例 Issue

```markdown
### Bug: Unable to connect to SQL database using Python

#### 問題描述 :
#### 在使用 Python 連接 MySQL 數據庫時，出現 `OperationalError`，提示無法連接到數據庫。

#### 實現步驟如下 :
1. 使用以下 Python 代碼嘗試連接數據庫：
   ```python
   import mysql.connector

   connection = mysql.connector.connect(
       host="localhost",
       user="username",
       password="password",
       database="test_db"
   )
2.運行代碼。
```

### 預期行為 :
應該成功連接到 MySQL 數據庫，並能夠進行數據查詢。

### 實際行為 :
運行代碼後，出現以下錯誤信息：

```rust
mysql.connector.errors.OperationalError: 1045 (28000): Access denied for user 'username'@'localhost'
```
### 環境 :
```
操作系統: Windows 10
Python 版本: 3.10
MySQL 版本: 8.0
```

## 提交 Issue
確保檢查你的描述是否清晰、完整，然後點擊「Submit new issue」提交。



