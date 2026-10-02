import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "RegLens",
  description:
    "Cited answers over RBI/SEBI regulation and Indian bank filings. Informational only, not legal or investment advice.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <header className="border-b border-[var(--border)] bg-[var(--surface)]">
          <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
            <Link href="/" className="text-lg font-semibold tracking-tight">
              RegLens
            </Link>
            <nav className="flex gap-6 text-sm text-[var(--muted)]">
              <Link href="/" className="hover:text-white">
                Status
              </Link>
              <Link href="/chat" className="hover:text-white">
                Ask
              </Link>
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-5xl px-6 py-8">{children}</main>
        {/* PRD non-goal #1: this notice is required on every surface. */}
        <footer className="mx-auto max-w-5xl px-6 pb-10 pt-4 text-xs text-[var(--muted)]">
          Informational only. Not legal or investment advice.
        </footer>
      </body>
    </html>
  );
}
