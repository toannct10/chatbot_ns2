// app/api/chat/route.ts
import { NextRequest, NextResponse } from "next/server";

const BACKEND = process.env.CHATBOT_BACKEND_URL;

export async function POST(req: NextRequest) {
  if (!BACKEND) {
    return NextResponse.json({ error: "Chatbot chưa cấu hình" }, { status: 500 });
  }
  try {
    const body = await req.json();
    const res = await fetch(`${BACKEND}/api/chat/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(35000), // Render free ngủ, wake up mất 30s
    });
    const data = await res.json();
    return NextResponse.json(data);
  } catch (err: any) {
    if (err.name === "TimeoutError") {
      // Trả về message thân thiện thay vì crash
      return NextResponse.json({
        answer: "⏳ Chatbot đang khởi động lại, vui lòng thử lại sau 30 giây.",
        sources: [],
        has_sources: false,
        follow_up_suggestions: [],
      });
    }
    return NextResponse.json({ error: "Lỗi kết nối chatbot" }, { status: 502 });
  }
}