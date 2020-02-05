#!/usr/bin/env python3

# IMPORTS
import sys
import time
import telebot
import yaml
from cobe.brain import Brain
from random import randint
import MySQLdb
from MySQLdb.constants import CR as MySQLdb_CR # https://github.com/PyMySQL/mysqlclient-python/blob/master/MySQLdb/constants/CR.py

# SETTINGS
settings = None
with open(r'settings.yml') as settings_yaml:
	settings = yaml.safe_load(settings_yaml)
	settings["bot_id"] = int(settings["bot_token"].split(':')[0])
	settings["chat_whitelist"][settings["creator_id"]] = settings["creator_nickname"]

# LOG STARTED
print("STARTED!")
print("Settings: {}".format(settings))

# CREATE OBJECTS
bot   = telebot.TeleBot(settings["bot_token"], threaded=False)
brain = None
while not brain:
	try:
		brain = Brain(settings["db_host"], settings["db_port"], settings["db_username"], settings["db_password"], settings["db_database"])
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
	print("-/getid {}: ".format(message.json), end='')
	if message.from_user.id == settings["creator_id"]:
		# Message is from the bot creator

		if message.reply_to_message:
			# Getid replying a message

			user_id         = message.reply_to_message.from_user.id
			user_first_name = message.reply_to_message.from_user.first_name

			# pyTelegramBotAPI doesn't support message.forward_sender_name yet. When supported, uncomment and replace "if message.reply_to_message.forward_from" with an elif.
			#if message.forward_sender_name:
			#	bot.reply_to( message, "Message forwarded from {}, which has ID {}.\nOriginal writer ({}) has hidden his ID.".format(user_first_name, user_id, message.forward_sender_name) )
			#	print("SENT USER ID, ORIGINAL WRITER HAS HIDDEN HIS ID")
			if message.reply_to_message.forward_from:
				bot.reply_to( message, "Message forwarded from {}, which has ID {}.\nOriginal writer ({}) has ID {}.".format(user_first_name, user_id, message.reply_to_message.forward_from.first_name, message.reply_to_message.forward_from.id) )
				print("SENT USER ID + FORWARD_FROM ID")
			else:
				bot.reply_to( message, "User ID of {} is {}".format(user_first_name, user_id) )
				print("SENT USER ID")

		else:
			# Getid received as a single message (not replying to another message)
			bot.reply_to( message, "Chat id: {}".format(message.chat.id) )
			print("SENT CHAT ID")

	else:
		# Message is not from the bot creator, so it's unauthorized
		bot.reply_to( message, "Only {} is allowed to use this command".format(settings["creator_nickname"]) )
		print("UNAUTHORIZED")

# ALL OTHER MESSAGES
@bot.message_handler(func=lambda message: True)
def all_messages(message):
	print("-Received {}: ".format(message.json), end='')
	if message.chat.id in settings["chat_whitelist"]:
		# Sender chat is in whitelist

		# Reply to private/group messages, trigger, random
		if ('group' in message.chat.type and \
                       ( (settings["trigger_enabled"] and start_with_trigger_words(message.text)) or (message.reply_to_message and settings["bot_id"]==message.reply_to_message.from_user.id) or (settings["random_enabled"] and randint(0,settings["random_percentage"])==0) ) \
                   ) \
                   or (message.chat.type=='private' and settings["private_reply_enabled"]):
			# Get reply from brain
			reply = brain.reply( remove_trigger_words(message.text) )
			# Remove backlisted words
			for word in settings["blacklist_words"]:
				# Replace is faster with an if-in check
				if word in reply:
					reply = reply.replace(word, '')
			# Capitalize reply
			reply = reply.lstrip().capitalize()
			# Log
			print("REPLIED '{}' ".format(reply), end='')
			# Send reply to telegram
			try:
				bot.reply_to( message, reply )
			except Exception as e:
				print("EXCEPTION: " + str(e))
		else:
			# Didn't replied
			print("REPLIED NOTHING ", end='')

		# If learn is enabled, then learn (but without trigger_words)
		if settings["learn_enabled"]:
			print("LEARNED", end='')
			brain.learn( remove_trigger_words(message.text) )
		else:
			print("NOT LEARNED", end='')

	else:
		# Sender chat is NOT in whitelist
		print("UNAUTHORIZED", end='')
		bot.reply_to( message, "This chat is not authorized: contact {}".format(settings["creator_nickname"]) )
	# Print newline into logs
	print()

# MESSAGES POLLING
bot.polling()
