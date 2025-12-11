# backend/main.py - DEPRECATED
# This file originally hosted a FastAPI application. The project now uses Django.
# If you need to restore FastAPI, retrieve the original file from version control history.
# Please run the Django server referenced in the README instead.
        
        # Step 4: Generate
        answer = chatbot.generate_response(request.question, chunks, request.history)
        
        return QueryResponse(answer=answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# Optional: Health check
@app.get("/health")
async def health_check():
    return {"status": "ok"}