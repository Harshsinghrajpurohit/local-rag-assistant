# Design Document: Visual and UX

This is a lightweight design document. The application is a local tool built with Streamlit, so visual design is minimal. Focus is on clarity, usability, and error handling.

## UI Layout (Streamlit)

### Overall Structure
- **Sidebar**: Document management (upload, list, delete)
- **Main area**: Chat interface (conversation history, input)
- **Expandable elements**: Source references for each answer

### Sidebar: Document Management

```
📚 Document Manager
─────────────────
📤 Upload PDFs
   [File uploader widget]
   
📋 Documents (3)
─────────────────
□ research_paper.pdf
  ✕ Delete
  
□ user_guide.pdf
  ✕ Delete
  
□ policies.pdf
  ✕ Delete
```

**Behavior**:
- File uploader accepts multiple PDFs
- Shows loading indicator while uploading
- Lists uploaded documents with file names
- Delete button requires confirmation ("Are you sure?")
- Shows clear error if upload fails
- Shows success message on completion

### Main Area: Chat Interface

```
Chat with Documents

User: What is the company's return policy?

🤖 Answer:
According to the policy document, returns are accepted within 
30 days of purchase with original receipt...

📌 Sources (2)
─────────────────
▶ policies.pdf, page 3
  "Returns are accepted within 30 days of purchase 
   with the original receipt, proof of payment, and 
   items in unused condition."

▶ policies.pdf, page 4
  "For damaged items, return shipping is covered by 
   the company..."

─────────────────
💬 Ask another question:
[Text input box]
```

**Behavior**:
- Chat history displayed chronologically
- User messages on right, AI responses on left
- Each AI response shows source documents in expandable section
- Sources show document name, page number, and exact passage
- Expandable/collapsible source details (collapsed by default)
- Loading indicator while generating answer
- Error message if query fails

### Visual Style (Streamlit Defaults)

- Use Streamlit's default theme (light/dark mode based on user preference)
- No custom CSS
- Clear typography (Streamlit defaults)
- Status messages in consistent format:
  - ✅ Success: "Uploaded successfully"
  - ⚠️ Warning: "No results found"
  - ❌ Error: "Failed to process file: [reason]"
  - ⏳ Loading: "Processing your question..."

## Error Handling and Messaging

### Document Upload Errors
```
❌ Upload Failed
File: document.pdf
Reason: File is too large (max 50 MB)
```

### Query Errors
```
❌ Query Failed
Error: Unable to generate answer. 
Details: Ollama service is not responding.
```

### Empty Results
```
⚠️ No Information Found
I searched the documents but didn't find enough information 
to answer your question. Try:
- Rephrasing your question
- Using different keywords
- Checking if the document covers this topic
```

### Missing Information (LLM Response)
```
I don't have enough information in the provided documents 
to answer this question accurately. The documents mention 
[related topic], but don't cover [your question].
```

## UX Principles

1. **Clarity Over Cleverness**: Use plain language, not jargon
2. **Transparency**: Show what's being searched and where results come from
3. **Feedback**: Always respond to user actions (success, error, waiting)
4. **Grounding**: Never show answers without source passages
5. **Simplicity**: One task per screen area
6. **Recovery**: Clear next steps when errors occur

## Conversation Management

- Conversation persists within a session (browser refresh keeps history)
- Option to clear conversation with confirmation
- Each conversation is isolated (separate from other users or sessions)
- No conversation history export (optional for Phase 9)

## Accessibility Considerations

- Use high contrast text (Streamlit defaults provide this)
- Expandable sections are keyboard navigable
- Error messages are clear and not relying on color alone
- Use semantic HTML structure (Streamlit provides this)

## Mobile Responsiveness

- Streamlit apps run in the browser
- Layout is responsive by default
- Sidebar collapses on small screens
- Chat messages stack vertically

## Performance and Loading States

```
⏳ Retrieving relevant documents...
   (Show progress, don't block UI)

⏳ Generating answer...
   (Show progress, don't block UI)

✅ Complete
```

- Show loading state immediately
- Update user on what's happening
- Estimated time if known (e.g., "Embedding PDF... 2-3 seconds")
- Allow cancel for long operations (if technically feasible)

## Configuration and Customization (Future)

- Settings page for:
  - Number of search results to retrieve
  - LLM temperature (determinism vs. creativity)
  - Chunk size and overlap
  - Embedding model selection

---

**Summary**: This is a functional, clear interface focused on answering questions grounded in documents. No fancy animations or advanced styling. The goal is usability and transparency, not visual complexity.
