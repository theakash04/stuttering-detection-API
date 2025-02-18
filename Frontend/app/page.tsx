'use client'

import { Button } from "@/components/ui/button";
import Link from "next/link";

const HeroSection = () => {
    return (
        <div className="min-w-screen min-h-screen bg-gradient-to-b from-background to-background px-4 py-12">
            <div className="max-w-7xl mx-auto">
                {/* Title Section */}
                <div className=" pb-8 mb-12 text-center">
                    <h1 className="font-extrabold md:text-8xl text-6xl bg-gradient-to-b from-sky-100 to-sky-200 bg-clip-text text-transparent tracking-tighter animate-shine">
                        VOCALS
                    </h1>
                </div>

                {/* One-Liner Description */}
                <p className="text-xl font-semibold text-sky-100 mb-16 text-center max-w-3xl mx-auto leading-relaxed">
                    Harness AI-powered insights to detect stuttering in real-time and unlock your confident communication.
                </p>

                <div className="mb-16 flex items-center justify-center">
                    <Link href="/detect">
                        <Button size={"lg"} className="font-bold">
                            Get Started
                        </Button>
                    </Link>
                </div>

                {/* Hero Highlights */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-24">
                    {[
                        {
                            title: "Real-Time Analysis",
                            desc: "Instant stuttering detection for immediate feedback",
                            icon: "graph.svg"
                        },
                        {
                            title: "Personalized Insights",
                            desc: "Tailored suggestions to improve speech fluency",
                            icon: "target.svg"
                        },
                        {
                            title: "Intuitive Interface",
                            desc: "Effortless navigation with modern design",
                            icon: "interface.svg"
                        },
                    ].map((feature, index) => (
                        <div
                            key={index}
                            className="bg-primary-foreground/70 p-8 rounded-2xl shadow-lg hover:shadow-xl transition-shadow duration-300 border border-transparent hover:border-blue-100"
                        >
                            <img className="mb-4 w-10 h-10" src={feature.icon} />
                            <h3 className="text-2xl font-bold text-sky-100 mb-3">{feature.title}</h3>
                            <p className="text-muted-foreground leading-relaxed">{feature.desc}</p>
                        </div>
                    ))}
                </div>

                {/* Footer */}
                <footer className="mt-24 pt-12 border-t border-slate-200">
                    <div className="text-center text-muted-foreground">
                        <p className="mb-4 font-bold">
                            &copy; WE TRIED OUR BEST!
                        </p>
                        <div className="space-x-6 flex items-center justify-center">
                            <a href="https://github.com/anubhavgh023" className="text-sky-100 transition-colors flex gap-2 hover:underline">
                                <p>ANUBHAV</p>
                            </a>
                            <a href="https://github.com/theakash04" className="text-sky-100 transition-colors flex gap-2 hover:underline">
                                <p>AKASH</p>
                            </a>
                            <a href="https://github.com/Rishabh-Gi-t" className="text-sky-100 transition-colors flex gap-2 hover:underline">
                                <p>RISHBH</p>
                            </a>
                            <a href="https://github.com/sobhitb033" className="text-sky-100 transition-colors flex gap-2 hover:underline">
                                <p>SOBHIT</p>
                            </a>
                            <a href="https://github.com/pranav5127" className="text-sky-100 transition-colors flex gap-2 hover:underline">
                                <p>PRANAV</p>
                            </a>
                        </div>
                    </div>
                </footer>
            </div>
        </div>);
};

export default HeroSection;

