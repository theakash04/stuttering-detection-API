import React, { useState, useEffect } from 'react';

interface TypewriterFeedbackProps {
    feedback?: string[];
}

const TypewriterFeedback: React.FC<TypewriterFeedbackProps> = ({ feedback = [] }) => {
    const [displayedFeedback, setDisplayedFeedback] = useState<string[]>([]);
    const [currentItemIndex, setCurrentItemIndex] = useState<number>(0);
    const [currentText, setCurrentText] = useState<string>('');
    const [isTyping, setIsTyping] = useState<boolean>(true);

    // Reset state when feedback changes
    useEffect(() => {
        setDisplayedFeedback([]);
        setCurrentItemIndex(0);
        setCurrentText('');
        setIsTyping(true);
    }, [feedback]);

    useEffect(() => {
        // Guard against undefined or empty feedback
        if (!feedback || !feedback.length) {
            setIsTyping(false);
            return;
        }

        if (currentItemIndex >= feedback.length) {
            setIsTyping(false);
            return;
        }

        const currentFeedbackItem = feedback[currentItemIndex];

        // Guard against invalid feedback item
        if (!currentFeedbackItem) {
            setIsTyping(false);
            return;
        }

        if (currentText.length < currentFeedbackItem.length) {
            const timeout = setTimeout(() => {
                setCurrentText(currentFeedbackItem.slice(0, currentText.length + 1));
            }, 20); // Adjust typing speed here (lower = faster)

            return () => clearTimeout(timeout);
        } else {
            const timeout = setTimeout(() => {
                setDisplayedFeedback((prev) => [...prev, currentText]);
                setCurrentText('');
                setCurrentItemIndex((prev) => prev + 1);
            }, 500); // Pause between items

            return () => clearTimeout(timeout);
        }
    }, [currentText, currentItemIndex, feedback]);

    if (!feedback || !feedback.length) {
        return null;
    }

    return (
        <div className="text-muted-foreground leading-relaxed">
            <div>
                {displayedFeedback.map((text, index) => (
                    <div key={index} className="mb-4">
                        <span>• {text}</span>
                    </div>
                ))}
                {isTyping && (
                    <div className="mb-4">
                        <span>• {currentText}</span>
                        <span className="animate-pulse">|</span>
                    </div>
                )}
            </div>
        </div>
    );
};

export default TypewriterFeedback;
