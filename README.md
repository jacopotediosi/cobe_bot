# Cobe_bot

## What and why
This project is a porting of the [Cobe Project](https://github.com/pteichman/cobe) (a popular conversation simulator based on MegaHAL), converted to Docker Compose, Python3 and MySQL, and connected to Telegram via the pyTelegramBotAPI library.

It entertained me and my classmates at the University of Milan for a few years, with two bots trained on our daily noob-slaying messages, simulating a communist user and a particularly arrogant computer security expert.

## Installation and configuration
1. Clone this repository.
2. Copy the `docker-compose.yml` file to `docker-compose.override.yml`, which will be your configuration file in your production environment. Since it is specified in the `.gitignore` file, there is no risk of this file being accidentally inserted into git.
3. Edit the settings (environment variables) in `docker-compose.override.yml` with yours:
	- `MYSQL_HOST`: This must contain the name of the container hosting the database, specified within the same file (e.g. `"db"`)
	- `MYSQL_PORT`: This must contain the port number on which to listen for the database, as a string (e.g. `"3306"`). Being an internal exposure only to the container group of this docker-compose, it is probably irrelevant to you.
	- `MYSQL_USERNAME`: This is the username used by all Cobe containers to access the database. There is currently no user separation implemented, so it must be `"root"` to work.
	- `MYSQL_ROOT_PASSWORD`: This is the database root password. Since the db is only exposed in the context of this docker-compose, it's probably not a security relevant parameter.
	- `MYSQL_DB`: This is the database name of the specific docker container. It should be a string.
	- `BOT_TOKEN`: Your Telegram bot token as a string (e.g. `"111111111:XXXX-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"`).
	- `CREATOR_ID`: Your Telegram ID as a string (e.g. `"00000000"`).
	- `CREATOR_NICKNAME`: Your Telegram nickname as a string. It is currently used by the bot only to print in chat who owns the bot.
	- `LEARN_ENABLED`: A string. If it is `"0"`, Cobe does not learn from the messages it receives. Any other value activates learning.
	- `CHAT_ALLOWLIST`: A string containing a comma-separated list of chat IDs where the bot is allowed to be added and used (e.g. `"00000000000,00000000001"`).
	- `TRIGGER_ENABLED`: If it is `"0"`, the bot does not respond to messages containing the trigger words indicated in the `TRIGGER_WORDS` environment variable. Any other value causes the bot to always reply to any message that contains a trigger word.
	- `TRIGGER_WORDS`: A string containing a comma-separated list of words (e.g. `"@cobe1bot,cobe1bot,cobe1"`) that, when contained in a message received by the bot, will cause a response from Cobe if `TRIGGER_ENABLED` is on.
	- `RANDOM_ENABLED`: If it is any string other than `"0"`, the bot replies to messages at an occasional rate, set by the `RANDOM_PERCENTAGE` environment variable.
	- `RANDOM_PERCENTAGE`: Every how many messages (approximately) the bot should respond if the `RANDOM_ENABLED` environment variable is on. It must be a string containing a number (e.g. `"200"`).
	- `PRIVATE_REPLY_ENABLED`: If it is any string other than `"0"`, the bot also responds to private chats as well as groups, only if the chat ID is contained in the `CHAT_ALLOWLIST` environment variable.
	- `BLOCKLIST_WORDS`: A string containing a comma-separated list of words that the bot should neither learn nor use (e.g., v`"#hashtag,bannedword"`).
4. You can also create multiple bots within a single docker-compose project, by simply creating multiple Cobe containers each with its own name and environment variables. The application is coded to create its own DB for each container by itself.
5. Build and run with `docker compose build && docker-compose up -d`.

## Know issues and to-do list
This project was a quick porting job done in a “just make it work” style, therefore over time a list of desired features developed, which were never implemented. Additionally, some unresolved bugs have been identified.
- Cobe was only converted syntactically without a detailed analysis of its functionality, which leaves the possibility of the existence of logical bugs.
- The application used to crash often with the error message `MySQL gone away`, so the entire database connection part should be reviewed.
- Even though the DBMS has been ported from SQLite to MySQL, the DB operations may not have been implemented in a thread-safe way. As a result, pyTelegramBotAPI was configured to enqueue received messages and work with only one thread. The entire threading part should be revisited.
- Only a few points in the code print logs using the Python `print()` function. It could be useful to revisit the logging as a whole, dividing logs into severity levels and ensuring that they are generated for every necessary operation.
- It has been noticed that Cobe seemed to handle accented words or apostrophes poorly. Specifically, the responses provided by the bot often used `e` instead of `è` and vice versa.
- If multiple Cobe containers are created in the same docker-compose (to create multiple bots while maintaining a single running MySQL container, as suggested in the installation instructions), all containers will connect to the database using the same root username and password. Therefore, the database user management should be improved and permissions should also be limited.
- It would be convenient if authorized users (the bot creator or administrators) could change the settings directly via chat commands.
