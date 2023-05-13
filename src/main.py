#!/usr/bin/env python3

# IMPORTS
import os
import sys
import time
import telebot
from cobe.brain import Brain
from random import randint
import MySQLdb
from MySQLdb.constants import CR as MySQLdb_CR # https://github.com/PyMySQL/mysqlclient-python/blob/master/MySQLdb/constants/CR.py

# SETTINGS
settings = {}
settings["mysql_host"]            = os.getenv("MYSQL_HOST")
settings["mysql_port"]            = int(os.getenv("MYSQL_PORT"))
settings["mysql_username"]        = os.getenv("MYSQL_USERNAME")
settings["mysql_root_password"]   = os.getenv("MYSQL_ROOT_PASSWORD")
settings["mysql_db"]              = os.getenv("MYSQL_DB")
settings["bot_token"]             = os.getenv("BOT_TOKEN")
settings["bot_id"]                = int(os.getenv("BOT_TOKEN").split(':')[0])
settings["creator_id"]            = int(os.getenv("CREATOR_ID"))
settings["creator_nickname"]      = os.getenv("CREATOR_NICKNAME")
settings["trigger_enabled"]       = bool(int(os.getenv("TRIGGER_ENABLED")))
settings["trigger_words"]         = os.getenv("TRIGGER_WORDS").split(",")
settings["chat_allowlist"]        = os.getenv("CHAT_ALLOWLIST").split(",")
settings["random_enabled"]        = bool(int(os.getenv("RANDOM_ENABLED")))
settings["random_percentage"]     = int(os.getenv("RANDOM_PERCENTAGE"))
settings["private_reply_enabled"] = bool(int(os.getenv("PRIVATE_REPLY_ENABLED")))
settings["blocklist_words"]       = os.getenv("BLOCKLIST_WORDS").split(",")
settings["learn_enabled"]         = bool(int(os.getenv("LEARN_ENABLED")))

# LOG STARTED
print("STARTED!")

# CREATE OBJECTS
bot   = telebot.TeleBot(settings["bot_token"], threaded=False)
brain = None
while not brain:
	try:
		brain = Brain(settings["mysql_host"], settings["mysql_port"], settings["mysql_username"], settings["mysql_root_password"], settings["mysql_db"])
	except MySQLdb._exceptions.OperationalError as e:
		if e.args[0] != MySQLdb_CR.CONN_HOST_ERROR:
			raise
		brain = None
		print("Waiting for DB to go UP...")
		time.sleep(20)

print("CONNECTED TO DB!")

# UTILITY FUNCTIONS
# Remove the first "trigger word" detected if string is beginning with it
def remove_trigger_words(string):
	trigger_word = start_with_trigger_words(string)
	if trigger_word:
		return string[len(trigger_word):]
	return string

# Return the "trigger word" the string is beginning with. None if not detected.
def start_with_trigger_words(string):
	for word in settings["trigger_words"]:
		if string[:len(word)].lower() == word.lower():
			return string[:len(word)]
	return None

# PROCESSING MESSAGES

# /getid
@bot.message_handler(commands=['getid'])
def get_id(message):
	# Log
	print(f"-/getid {message.json}: ", end='')
	if message.from_user.id == settings["creator_id"]:
		# Message is from the bot creator

		if message.reply_to_message:
			# Getid replying a message

			user_id         = message.reply_to_message.from_user.id
			user_first_name = message.reply_to_message.from_user.first_name

			# pyTelegramBotAPI doesn't support message.forward_sender_name yet. When supported, uncomment and replace "if message.reply_to_message.forward_from" with an elif.
			#if message.forward_sender_name:
			#	bot.reply_to(message, f"Message forwarded from {user_first_name}, which has ID {user_id}.\nOriginal writer ({message.forward_sender_name}) has hidden his ID.")
			#	print("SENT USER ID, ORIGINAL WRITER HAS HIDDEN HIS ID")
			if message.reply_to_message.forward_from:
				bot.reply_to(message, f"Message forwarded from {user_first_name}, which has ID {user_id}.\nOriginal writer ({message.reply_to_message.forward_from.first_name}) has ID {message.reply_to_message.forward_from.id}.")
				print("SENT USER ID + FORWARD_FROM ID")
			else:
				bot.reply_to(message, f"User ID of {user_first_name} is {user_id}")
				print("SENT USER ID")

		else:
			# Getid received as a single message (not replying to another message)
			bot.reply_to(message, f"Chat id: {message.chat.id}")
			print("SENT CHAT ID")

	else:
		# Message is not from the bot creator, so it's unauthorized
		bot.reply_to(message, f"Only {settings['creator_nickname']} is allowed to use this command")
		print("UNAUTHORIZED")

# ALL OTHER MESSAGES
@bot.message_handler(func=lambda message: True)
def all_messages(message):
	print(f"-Received {message.json}: ", end='')
	if str(message.chat.id) in settings["chat_allowlist"] or message.chat.id==settings["creator_id"]:
		# Sender chat is in whitelist

		# Reply to private/group messages, trigger, random
		if ('group' in message.chat.type and \
                       (settings["trigger_enabled"] and start_with_trigger_words(message.text)) or (message.reply_to_message and settings["bot_id"]==message.reply_to_message.from_user.id) or (settings["random_enabled"] and randint(0, settings["random_percentage"])==0) \
                   ) \
                   or (message.chat.type=='private' and settings["private_reply_enabled"]):
			# Get reply from brain
			reply = brain.reply(remove_trigger_words(message.text))
			# Remove blocklisted words
			for word in settings["blocklist_words"]:
				# Replace is faster with an if-in check
				if word in reply:
					reply = reply.replace(word, '')
			# Capitalize reply
			reply = reply.lstrip().capitalize()
			# Log
			print(f"REPLIED '{reply}' ", end='')
			# Send reply to telegram
			try:
				bot.reply_to(message, reply)
			except Exception as e:
				print(f"EXCEPTION: {str(e)}")
		else:
			# Didn't replied
			print("REPLIED NOTHING ", end='')

		# If learn is enabled, then learn (but without trigger_words)
		if settings["learn_enabled"]:
			print("LEARNED", end='')
			brain.learn(remove_trigger_words(message.text))
		else:
			print("NOT LEARNED", end='')

	else:
		# Sender chat is NOT in allowlist
		print("UNAUTHORIZED", end='')
		bot.reply_to(message, f"This chat is not authorized: contact {settings['creator_nickname']}")
	# Print newline into logs
	print()

# MESSAGES POLLING
bot.polling()
