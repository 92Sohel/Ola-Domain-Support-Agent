"""
multimodal_support.py - Multimodal Document & Image Analysis for Ola Domain Support Agent
Track: Business Operations / Customer Support (Ola)

Provides:
1. PDF Text & Metadata Extraction (via pypdf & pdfminer)
2. Image OCR & Analysis (via Windows Native Media OCR & PIL)
3. Support Pipeline Integration: Feeds extracted text & user prompt into the Ola Support Agent
4. FastAPI Router (POST /multimodal/analyze) for multipart file uploads
5. Standalone CLI support for direct terminal analysis
"""

import io
import os
import re
import sys
import uuid
import argparse
import subprocess
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from pydantic import BaseModel, Field
from PIL import Image

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

# FastAPI imports for optional route mounting
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from app import ask_endpoint, AskRequest, AskResponse


# ==========================================
# 1. PDF Parser
# ==========================================

def extract_text_from_pdf(file_bytes_or_path) -> Tuple[str, Dict[str, Any]]:
    """
    Extracts text and metadata from a PDF file using pypdf with pdfminer fallback.
    Returns: (extracted_text, metadata_dict)
    """
    text_content = []
    metadata = {"pages": 0, "parser": "none"}
    
    if isinstance(file_bytes_or_path, (str, Path)):
        stream = open(file_bytes_or_path, "rb")
    elif isinstance(file_bytes_or_path, bytes):
        stream = io.BytesIO(file_bytes_or_path)
    else:
        stream = file_bytes_or_path

    # Attempt with pypdf first
    if PYPDF_AVAILABLE:
        try:
            reader = PdfReader(stream)
            metadata["pages"] = len(reader.pages)
            metadata["parser"] = "pypdf"
            
            # Extract doc info if present
            if reader.metadata:
                metadata["title"] = reader.metadata.title or ""
                metadata["author"] = reader.metadata.author or ""
                
            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_content.append(f"--- Page {idx + 1} ---\n{page_text.strip()}")
        except Exception as e:
            text_content = []
            
    # Fallback to pdfminer if pypdf yielded no text
    full_text = "\n\n".join(text_content).strip()
    if not full_text and PDFMINER_AVAILABLE:
        try:
            if hasattr(stream, "seek"):
                stream.seek(0)
            miner_text = pdfminer_extract_text(stream)
            if miner_text and miner_text.strip():
                full_text = miner_text.strip()
                metadata["parser"] = "pdfminer"
        except Exception:
            pass
            
    return full_text, metadata


# ==========================================
# 2. Image OCR & Analysis Parser
# ==========================================

def ocr_image_windows_native(image_path_or_bytes) -> str:
    """
    Performs local, zero-network OCR on images using Windows built-in
    Windows.Media.Ocr runtime engine.
    """
    temp_path = None
    if isinstance(image_path_or_bytes, (str, Path)):
        resolved_path = os.path.abspath(image_path_or_bytes)
    else:
        temp_path = os.path.abspath(f"temp_ocr_{uuid.uuid4().hex[:8]}.png")
        if isinstance(image_path_or_bytes, bytes):
            with open(temp_path, "wb") as f:
                f.write(image_path_or_bytes)
        else:
            image_path_or_bytes.save(temp_path)
        resolved_path = temp_path

    ps_script = f"""
    Add-Type -AssemblyName System.Runtime.WindowsRuntime
    $asTaskGeneric = [System.WindowsRuntimeSystemExtensions].GetMethods() | ? {{ $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' }}[0]

    Function Await($WinRtTask, $ResultType) {{
        $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
        $netTask = $asTask.Invoke($null, @($WinRtTask))
        $netTask.Wait(-1) | Out-Null
        $netTask.Result
    }}

    [Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime] | Out-Null
    [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType=WindowsRuntime] | Out-Null
    [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime] | Out-Null

    $target = [System.IO.Path]::GetFullPath('{resolved_path}')
    if (-not (Test-Path $target)) {{ exit 0 }}

    try {{
        $file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($target)) ([Windows.Storage.StorageFile])
        $stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
        $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
        $bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])

        $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
        if (-not $engine) {{
            $lang = [Windows.Globalization.Language]::new('en-US')
            $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang)
        }}
        $result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
        Write-Output $result.Text
    }} catch {{
        # Graceful return if OCR fails
    }}
    """
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=15
        )
        output_text = proc.stdout.strip()
    except Exception:
        output_text = ""
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    return output_text


def extract_content_from_image(file_bytes_or_path) -> Tuple[str, Dict[str, Any]]:
    """
    Inspects image properties and extracts any printed text via native OCR.
    Returns: (extracted_text, metadata_dict)
    """
    if isinstance(file_bytes_or_path, (str, Path)):
        img = Image.open(file_bytes_or_path)
        img_source = file_bytes_or_path
    elif isinstance(file_bytes_or_path, bytes):
        img = Image.open(io.BytesIO(file_bytes_or_path))
        img_source = file_bytes_or_path
    else:
        img = Image.open(file_bytes_or_path)
        img_source = file_bytes_or_path

    metadata = {
        "format": img.format,
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
    }

    # Perform OCR
    recognized_text = ocr_image_windows_native(img_source)
    
    return recognized_text, metadata


# ==========================================
# 3. Unified Document & Multimodal Analyzer
# ==========================================

class MultimodalAnalysisResult(BaseModel):
    filename: str = Field(description="Name of the uploaded file")
    file_type: str = Field(description="Detected type: 'pdf' or 'image'")
    extracted_text: str = Field(description="Raw text extracted via parser or OCR")
    detected_entities: Dict[str, List[str]] = Field(default_factory=dict, description="Extracted ticket IDs, amounts, dates")
    user_prompt: Optional[str] = Field(default=None, description="Optional user prompt provided with file")
    agent_response: AskResponse = Field(description="Full structured response from Ola Domain Support Agent")


def extract_entities_from_text(text: str) -> Dict[str, List[str]]:
    """Finds Ola ticket IDs, CRN booking numbers, and currency amounts."""
    entities = {}
    
    tickets = re.findall(r'OLA-TCK-\d{4}', text, re.IGNORECASE)
    if tickets:
        entities["ticket_ids"] = list(dict.fromkeys([t.upper() for t in tickets]))
        
    crns = re.findall(r'CRN[-_]?\d{7,10}', text, re.IGNORECASE)
    if crns:
        entities["booking_crns"] = list(dict.fromkeys([c.upper() for c in crns]))
        
    amounts = re.findall(r'[₹Rs\.]\s*\d+(?:,\d+)*(?:\.\d{2})?', text)
    if amounts:
        entities["amounts"] = list(dict.fromkeys(amounts))
        
    return entities


def analyze_file_and_query_agent(
    file_bytes_or_path,
    filename: str,
    user_prompt: Optional[str] = None,
    session_id: str = "multimodal-session"
) -> MultimodalAnalysisResult:
    """
    1. Detects file type (PDF or Image)
    2. Extracts text content & metadata
    3. Detects key domain entities (ticket IDs, ride CRNs)
    4. Composes an operational query and executes through Ola Domain Support Agent
    5. Returns unified MultimodalAnalysisResult
    """
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == ".pdf":
        file_type = "pdf"
        extracted_text, meta = extract_text_from_pdf(file_bytes_or_path)
    elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
        file_type = "image"
        extracted_text, meta = extract_content_from_image(file_bytes_or_path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Supported formats: .pdf, .png, .jpg, .jpeg, .webp")

    detected_entities = extract_entities_from_text(extracted_text)
    
    # Formulate query for the support agent
    query_parts = []
    if user_prompt and user_prompt.strip():
        query_parts.append(user_prompt.strip())
        
    if detected_entities.get("ticket_ids"):
        tid = detected_entities["ticket_ids"][0]
        query_parts.append(f"Regarding support ticket {tid}")
        
    if extracted_text.strip():
        # Truncate text context to first 300 chars to avoid prompt bloat
        snippet = extracted_text.strip()[:300].replace("\n", " ")
        query_parts.append(f"(Extracted from {filename}: \"{snippet}\")")
    elif not user_prompt:
        query_parts.append(f"Please inspect and summarize the attached {file_type} document '{filename}' for Ola support policy guidance.")

    synthesized_query = " ".join(query_parts) if query_parts else f"Review uploaded document {filename}"
    
    # Run through the Ola Domain Support Agent
    ask_req = AskRequest(query=synthesized_query, session_id=session_id)
    agent_resp = ask_endpoint(ask_req)
    
    return MultimodalAnalysisResult(
        filename=filename,
        file_type=file_type,
        extracted_text=extracted_text,
        detected_entities=detected_entities,
        user_prompt=user_prompt,
        agent_response=agent_resp
    )


# ==========================================
# 4. FastAPI Router (Mountable in app.py)
# ==========================================

multimodal_router = APIRouter(prefix="/multimodal", tags=["Multimodal Document Support"])

@multimodal_router.post(
    "/analyze",
    response_model=MultimodalAnalysisResult,
    summary="Upload and analyze a PDF receipt or ride screenshot"
)
async def analyze_uploaded_file(
    file: UploadFile = File(..., description="PDF document or image file (.pdf, .png, .jpg, .webp)"),
    prompt: Optional[str] = Form(None, description="Optional user question regarding this file"),
    session_id: Optional[str] = Form("multimodal-session", description="Conversation session ID")
):
    """
    Accepts a PDF document or image file, performs local OCR/text extraction,
    and runs the synthesized inquiry through the Ola Domain Support Agent multi-agent pipeline.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty."
        )
        
    try:
        result = analyze_file_and_query_agent(
            file_bytes_or_path=contents,
            filename=file.filename or "uploaded_file",
            user_prompt=prompt,
            session_id=session_id
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Multimodal processing error: {str(e)}")


# ==========================================
# 5. Standalone CLI Interface
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ola Multimodal Document & Image Analyzer")
    parser.add_argument("--file", required=True, help="Path to PDF or image file")
    parser.add_argument("--prompt", default=None, help="Optional user question about the document")
    parser.add_argument("--session", default="cli-multimodal", help="Session ID for chat memory")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found at '{args.file}'")
        sys.exit(1)

    print("=" * 70)
    print(f"Analyzing File: {os.path.basename(args.file)}")
    print("=" * 70)
    
    res = analyze_file_and_query_agent(
        file_bytes_or_path=args.file,
        filename=os.path.basename(args.file),
        user_prompt=args.prompt,
        session_id=args.session
    )
    
    print(f"\n[File Info]")
    print(f"  Type:           {res.file_type.upper()}")
    print(f"  Detected Items: {res.detected_entities}")
    print(f"\n[Extracted Text Content]")
    print(f"  {res.extracted_text[:400] if res.extracted_text else '(No text recognized)'}")
    print("\n" + "=" * 70)
    print("[Agent Policy / Ticket Response]")
    print("=" * 70)
    print(f"  Response Type: {res.agent_response.response_type.upper()}")
    print(f"  Sources:       {', '.join(res.agent_response.sources)}")
    print(f"  Status:        {'APPROVED' if res.agent_response.autogen_approved else 'FLAGGED'}")
    print(f"\n  {res.agent_response.answer}\n")
    print("=" * 70)
