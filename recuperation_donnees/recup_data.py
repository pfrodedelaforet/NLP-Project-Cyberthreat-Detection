
import tweepy as tw
import xlsxwriter
import openpyxl
workbook = xlsxwriter.Workbook('tablerase.xlsx')
worksheet = workbook.add_worksheet()
with open('cles.txt', 'r') as f:
    lignes = []
    for line in f:
        l = line.strip().split(' : ')
        lignes.append(l)
credentials = {}
credentials[lignes[0][0]] = lignes[0][1]
credentials[lignes[1][0]] = lignes[1][1]
credentials[lignes[2][0]] = lignes[2][1]
credentials[lignes[3][0]] = lignes[3][1]

auth = tw.OAuthHandler(credentials['CONSUMER_KEY'], credentials['CONSUMER_SECRET'])
auth.set_access_token(credentials['ACCESS_TOKEN'], credentials['ACCESS_SECRET'])

#Get the ids of the studied twitter accounts
with open('twitter_channels.txt', 'r') as f:
    liste_liens = f.readlines()
    usernames = []
for k in range(len(liste_liens)) :
    if liste_liens[k][0:20] == 'https://twitter.com/':
        usernames.append(liste_liens[k][20:len(liste_liens[k])-1])
api = tw.API(auth, wait_on_rate_limit=True)
j = 1
for i in range(len(usernames)):
    print(i)
    for tweet in api.user_timeline(screen_name = usernames[i], count = 200):#can't have more than 200 by user
        if tweet.entities['urls'] != [] :
            j += 1
            worksheet.write(j, 0, str(tweet.id))
            worksheet.write(j, 1, tweet.user.screen_name)
            worksheet.write(j, 2, str(tweet.created_at))
            worksheet.write(j, 3, tweet.entities['urls'][0]['expanded_url'])
            worksheet.write(j, 4, tweet.retweet_count)
            worksheet.write(j, 5, tweet.favorite_count)
            worksheet.write(j, 6, tweet.text)

workbook.close()



