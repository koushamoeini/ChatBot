import os
import json
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from backend.chatbot import RAGChatbot


# Lazy-load and require API_KEY from environment for security
chatbot = None
def get_chatbot():
    global chatbot
    if chatbot is None:
        api_key = os.environ.get('API_KEY')
        if not api_key:
            raise RuntimeError('API_KEY environment variable is not set. Please set it before running the server.')
        chatbot = RAGChatbot(api_key=api_key)
    return chatbot


def index(request):
    return render(request, 'index.html')


@csrf_exempt
def ask(request):
    if request.method != 'POST':
        return JsonResponse({'detail': 'Method not allowed'}, status=405)

    try:
        body = json.loads(request.body)
        question = body.get('question', '')
        history = body.get('history', []) or []

        bot = get_chatbot()
        standalone = bot.rewrite_query(question, history)
        queries = bot.expand_query(standalone)
        chunks = bot.retrieve_relevant_chunks(queries)
        answer = bot.generate_response(question, chunks, history)

        return JsonResponse({'answer': answer})
    except Exception as e:
        return JsonResponse({'detail': f'Internal error: {str(e)}'}, status=500)
